"""Relax impulse locations by inserting free interior position nodes.

Given the encounter order/epochs, an extra maneuver is permitted halfway along
each coast. Free-node positions are optimized using locally sparse derivatives.
Targets remain exact boundary conditions, without velocity-matching constraints.
"""

import json
import math
import time
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint
from .physics import MU, RE, norm, target_state, lambert, arc_min_radius
from .advanced_search import route_cache, route_x, initial_from_x


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
def positions_value_gradient(x, positions, times, branches, vstart, free_indices, smoothing):
    n = len(times) - 1
    r = positions.copy()
    for j in range(len(free_indices)):
        r[free_indices[j]] = x[j * 3 : j * 3 + 3] * 30000.0
    vi = np.empty((n, 3))
    vf = np.empty((n, 3))
    pen = np.empty(n)
    cost = 0.0
    for i in range(n):
        vi[i], vf[i], pen[i] = segment(r[i], r[i + 1], times[i + 1] - times[i], branches[i])
    for i in range(n):
        cost += impulse_cost(vi[i] - (vstart if i == 0 else vf[i - 1]), smoothing) + pen[i]
    gradient = np.zeros_like(x)
    step = 1e-06
    for j in range(len(free_indices)):
        k = free_indices[j]
        for axis in range(3):
            values = np.zeros(2)
            for si in range(2):
                rr = r[k].copy()
                rr[axis] += (1 if si == 0 else -1) * step * 30000
                a, z, p = segment(r[k - 1], rr, times[k] - times[k - 1], branches[k - 1])
                b, w, q = segment(rr, r[k + 1], times[k + 1] - times[k], branches[k])
                value = (
                    impulse_cost(a - (vstart if k == 1 else vf[k - 2]), smoothing)
                    + impulse_cost(b - z, smoothing)
                    + p
                    + q
                )
                if k < n - 1:
                    value += impulse_cost(vi[k + 1] - w, smoothing)
                values[si] = value
            gradient[j * 3 + axis] = (values[0] - values[1]) / (2 * step)
    return (cost, gradient)


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
def joint_value_gradient(x, states, ids, branches, free, smooth, want_gradient=True):
    n = len(ids) - 1
    nf = len(free)
    off = 4 + 3 * nf
    initial = initial_from_x(x)
    vstart = initial[3:]
    t = np.zeros(n + 1)
    t[1:] = x[off:] * 3600
    r = np.zeros((n + 1, 3))
    r[0] = initial[:3]
    for i in range(nf):
        r[free[i]] = x[4 + 3 * i : 7 + 3 * i] * 30000
    for k in range(1, n + 1):
        if ids[k] >= 0:
            r[k] = target_state(states, ids[k], t[k])[0]
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
        h = 1e-06 if j >= off else 1e-07
        values = np.zeros(2)
        for si in range(2):
            delta = h * (1 if si == 0 else -1)
            if j < 4:
                xx = x[:4].copy()
                xx[j] += delta
                init = initial_from_x(xx)
                values[si] = node_local(0, init[:3], 0.0, r, t, vi, vf, branches, init[3:], smooth)
            elif j < off:
                k = free[(j - 4) // 3]
                rr = r[k].copy()
                rr[(j - 4) % 3] += delta * 30000
                values[si] = node_local(k, rr, t[k], r, t, vi, vf, branches, vstart, smooth)
            else:
                k = j - off + 1
                tt = t[k] + delta * 3600
                rr = target_state(states, ids[k], tt)[0] if ids[k] >= 0 else r[k]
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
    from .search import Route

    for key in ["encounter_offsets_km", "encounter_ball_history", "encounter_ball_radius_km"]:
        data.pop(key, None)
    route = Route(**data["source_route"])
    route.initial = data["initial"]
    ids = np.array(data["node_ids"], dtype=np.int64)
    free = np.where(ids[1:] < 0)[0] + 1
    branches = np.array(data["branches"], dtype=np.int64)
    x = np.r_[
        route_x(route)[:4],
        np.array(data["positions"])[free].ravel() / 30000,
        np.array(data["times"])[1:] / 3600,
    ]
    off = 4 + 3 * len(free)
    n = len(ids) - 1
    matrix = np.zeros((n, len(x)))
    for k in range(n):
        matrix[k, off + k] = 1
        if k:
            matrix[k, off + k - 1] = -1
    bounds = (
        [(None, None), (-1.5707, 1.5707), (None, None), (-0.999999, 0.999999)]
        + [(None, None)] * (3 * len(free))
        + [(min_gap_hours, 240)] * n
    )
    best = [joint_value_gradient(x, states, ids, branches, free, 0.0)[0], x.copy()]
    start = time.monotonic()
    deadline = start + minutes * 60
    calls = [0]
    for smoothing in smoothings:

        def fun(z):
            if time.monotonic() > deadline:
                raise TimeoutError
            c, g = joint_value_gradient(z, states, ids, branches, free, smoothing)
            calls[0] += 1
            if calls[0] % 100 == 0:
                print("JOINT", calls[0], c, "smooth", smoothing, flush=True)
            raw = joint_value_gradient(z, states, ids, branches, free, 0.0, False)[0]
            if np.min(np.diff(np.r_[0, z[off:]])) >= min_gap_hours - 1e-09 and raw < best[0]:
                best[:] = [raw, z.copy()]
            return (c, g)

        try:
            if method == "L-BFGS-B":
                epochs = np.r_[0, x[off:], 240.05]
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
                    bounds=bounds[:off] + temporal,
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
    times = np.r_[0, x[off:] * 3600]
    initial = initial_from_x(x)
    pos = np.array(data["positions"])
    pos[0] = initial[:3]
    pos[free] = x[4:off].reshape(-1, 3) * 30000
    for k in range(1, len(ids)):
        if ids[k] >= 0:
            pos[k] = target_state(states, ids[k], times[k])[0]
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
    """Subdivide long-angle coasts to stabilize ATK's J2 Lambert corrector.

    The inserted midpoint lies on the very same Kepler arc and adds zero
    screening impulse; ATK then corrects each shorter arc independently.
    """
    import copy

    data = copy.deepcopy(data)
    pos = [data["positions"][0]]
    times = [0.0]
    ids = [-1]
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
            ids.append(k)
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
        node_ids=ids,
        branches=branches,
        subdivision_max_angle_deg=max_angle,
    )
    return data


def split_route(route, states, fraction=0.5):
    pos, vin, vout = route_cache(route, states)
    positions = [pos[0]]
    times = [0.0]
    ids = [-1]
    branches = []
    free = []

    def rhs(t, y):
        return np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]

    for i in range(len(route.ids)):
        t0 = 0 if i == 0 else route.times[i - 1]
        duration = route.times[i] - t0
        mid = solve_ivp(
            rhs,
            [0, duration * fraction],
            np.r_[pos[i], vout[i]],
            method="DOP853",
            rtol=1e-11,
            atol=1e-10,
        ).y[:, -1]
        points = [mid[:3], pos[i + 1]]
        vexpected = [vout[i], mid[3:]]
        periods = [duration * fraction, duration * (1 - fraction)]
        for point, velocity, dt in zip(points, vexpected, periods):
            choices = []
            for rev in range(3):
                for major in range(2):
                    for side in range(1 if rev == 0 else 2):
                        a, z = lambert(positions[-1], point, dt, major, rev, side)
                        if np.isfinite(a[0]):
                            choices.append((norm(a - velocity), [major, rev, side]))
            error, b = min(choices, key=lambda p: p[0])
            if error > 1e-05:
                raise ValueError("Could not identify split Lambert arc")
            branches.append(b)
            positions.append(point)
        times.extend([t0 + duration * fraction, route.times[i]])
        ids.extend([-1, route.ids[i]])
        free.append(2 * i + 1)
    return (np.array(positions), np.array(times), np.array(branches), np.array(free), ids)
