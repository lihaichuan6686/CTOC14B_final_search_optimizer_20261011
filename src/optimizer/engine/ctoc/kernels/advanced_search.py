"""Fuel-focused smooth continuation and destroy/repair search.

Keeps actual arrival velocity. Every accepted complete route still requires ATK
J2 targeting, saved-scene replay and the official checker before publication.
"""

import math
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize, minimize_scalar, LinearConstraint
from .physics import MU, RE, norm, target_state, lambert, arc_min_radius
from .search import Route, chain_cost


@njit(cache=True)
def initial_from_x(x):
    lon, lat, heading, altitude = (x[0], x[1], x[2], 600 + 10 * x[3])
    unit = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(unit, east)
    v = math.sqrt(MU / (RE + altitude)) * (math.cos(heading) * east + math.sin(heading) * north)
    return np.concatenate(((RE + altitude) * unit, v))


def route_x(route):
    r = np.array(route.initial[:3])
    v = np.array(route.initial[3:])
    radius = norm(r)
    lon = np.arctan2(r[1], r[0])
    lat = np.arcsin(r[2] / radius)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(r / radius, east)
    heading = np.arctan2(np.dot(v, north), np.dot(v, east))
    return np.r_[lon, lat, heading, (radius - RE - 600) / 10, np.array(route.times) / 3600]


@njit(cache=True)
def smooth_value(x, states, ids, branches, smoothing):
    initial = initial_from_x(x)
    c, dv, _, _ = chain_cost(states, ids, x[4:] * 3600, branches, initial)
    if c > 10000.0:
        return c
    for i in range(len(ids)):
        c += math.sqrt(np.dot(dv[i], dv[i]) + smoothing * smoothing) - norm(dv[i])
    return c


@njit(cache=True)
def smooth_value_gradient(x, states, ids, branches, smoothing):
    c = smooth_value(x, states, ids, branches, smoothing)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        step = 1e-05 if i >= 4 else 1e-06
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        grad[i] = (
            smooth_value(xp, states, ids, branches, smoothing)
            - smooth_value(xm, states, ids, branches, smoothing)
        ) / (2 * step)
    return (c, grad)


def polish(
    route, states, maxiter=200, deadline=None, smoothings=(0.03, 0.003, 0.0001), window=None
):
    ids = np.array(route.ids, dtype=np.int64)
    branches = np.array(route.branches, dtype=np.int64)
    x = route_x(route)
    best = route
    mat = np.zeros((len(ids), len(x)))
    for i in range(len(ids)):
        mat[i, i + 4] = 1
        if i:
            mat[i, i + 3] = -1
    bounds = [
        (None, None),
        (-np.pi / 2 + 1e-05, np.pi / 2 - 1e-05),
        (None, None),
        (-0.999999, 0.999999),
    ]
    bounds += [(0.05, 240.0) for _ in ids]
    if window is not None:
        bounds[4:] = [(max(0.05, t - window), min(240.0, t + window)) for t in x[4:]]
    for smoothing in smoothings:

        def fun(z):
            if deadline and time.monotonic() > deadline:
                raise TimeoutError
            return smooth_value_gradient(z, states, ids, branches, smoothing)

        try:
            result = minimize(
                fun,
                x,
                jac=True,
                method="SLSQP",
                bounds=bounds,
                constraints=[LinearConstraint(mat, 0.05, np.inf)],
                options={"maxiter": maxiter, "ftol": 1e-09},
            )
            x = result.x
        except TimeoutError:
            break
        c, _, vf, mr = chain_cost(states, ids, x[4:] * 3600, branches, initial_from_x(x))
        if mr >= RE + 205 and np.min(np.diff(np.r_[0, x[4:]])) >= 0.05 - 1e-08 and (c < best.cost):
            best = Route(
                route.ids,
                (x[4:] * 3600).tolist(),
                route.branches,
                initial_from_x(x).tolist(),
                float(c),
                vf.tolist(),
            )
    return best


BRANCHES = np.array(
    [[m, r, s] for r in range(3) for m in range(2) for s in range(1 if r == 0 else 2)],
    dtype=np.int64,
)


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


def route_cache(route, states):
    pos = [np.array(route.initial[:3])]
    vin = [np.array(route.initial[3:])]
    vout = []
    t0 = 0.0
    for k, t, b in zip(route.ids, route.times, route.branches):
        r, _ = target_state(states, k, t)
        a, z = lambert(pos[-1], r, t - t0, *b)
        pos.append(r)
        vout.append(a)
        vin.append(z)
        t0 = t
    return (np.array(pos), np.array(vin), np.array(vout))


def evaluated(ids, times, branches, initial, states):
    c, _, vf, mr = chain_cost(
        states,
        np.array(ids, dtype=np.int64),
        np.array(times),
        np.array(branches, dtype=np.int64),
        np.array(initial),
    )
    return Route(
        list(ids), list(times), np.asarray(branches).tolist(), list(initial), float(c), vf.tolist()
    )


def remove_node(route, index, states):
    ids = route.ids.copy()
    times = route.times.copy()
    branches = route.branches.copy()
    ids.pop(index)
    times.pop(index)
    branches.pop(index)
    if index < len(ids):
        pos, vin, vout = route_cache(route, states)
        t0 = 0 if index == 0 else route.times[index - 1]
        ai, af = arc_choices(pos[index], pos[index + 2], route.times[index + 1] - t0, 2)
        costs = np.linalg.norm(ai - vin[index], axis=1)
        if index + 2 < len(route.ids):
            costs += np.linalg.norm(vout[index + 2] - af, axis=1)
        costs = np.nan_to_num(costs, nan=1000000000.0)
        branches[index] = BRANCHES[np.argmin(costs)].tolist()
    return evaluated(ids, times, branches, route.initial, states)


@njit(cache=True)
def insertion_value(t, k, t0, t1, r0, r1, vprev, vnext, has_next, states, maxrev):
    rk, _ = target_state(states, k, t)
    a, z = arc_choices(r0, rk, t - t0, maxrev)
    best = 1000000000.0
    ibest = 0
    jbest = 0
    if t1 < 0:
        for i in range(len(BRANCHES)):
            if not np.isfinite(a[i, 0]):
                continue
            c = norm(a[i] - vprev)
            if c < best:
                best = c
                ibest = i
        return (best, ibest, jbest)
    b, w = arc_choices(rk, r1, t1 - t, maxrev)
    for i in range(len(BRANCHES)):
        if not np.isfinite(a[i, 0]):
            continue
        c0 = norm(a[i] - vprev)
        for j in range(len(BRANCHES)):
            if not np.isfinite(b[j, 0]):
                continue
            c = c0 + norm(b[j] - z[i])
            if has_next:
                c += norm(vnext - w[j])
            if c < best:
                best = c
                ibest = i
                jbest = j
    return (best, ibest, jbest)


@njit(cache=True)
def insertion_grid(ids, times, pos, vin, vout, k, states, step, maxrev):
    n = len(ids)
    records = np.full((n + 1, 5), 1000000000.0)
    for j in range(n + 1):
        t0 = 0.0 if j == 0 else times[j - 1]
        t1 = times[j] if j < n else -1.0
        upper = times[j] if j < n else 864000.0
        r1 = pos[j + 1] if j < n else np.zeros(3)
        has_next = j + 1 < n
        vnext = vout[j + 1] if has_next else np.zeros(3)
        old = norm(vout[j] - vin[j]) if j < n else 0.0
        if has_next:
            old += norm(vout[j + 1] - vin[j + 1])
        count = max(3, int((upper - t0) / step) + 1)
        for q in range(count):
            t = t0 + 181.0 + (upper - t0 - 362.0) * q / (count - 1)
            if t <= t0 or t >= upper:
                continue
            c, i1, i2 = insertion_value(
                t, k, t0, t1, pos[j], r1, vin[j], vnext, has_next, states, maxrev
            )
            c -= old
            if c < records[j, 0]:
                records[j] = np.array([c, j, t, i1, i2])
    return records


def insert_node(route, k, states, rng=None, step=1200, maxrev=1, alternatives=6):
    pos, vin, vout = route_cache(route, states)
    records = insertion_grid(
        np.array(route.ids), np.array(route.times), pos, vin, vout, k, states, step, maxrev
    )
    candidates = []
    for rec in sorted(records, key=lambda r: r[0])[:alternatives]:
        _, j, t, _, _ = rec
        j = int(j)
        t0 = 0 if j == 0 else route.times[j - 1]
        t1 = route.times[j] if j < len(route.ids) else -1
        upper = t1 if t1 >= 0 else 864000
        r1 = pos[j + 1] if j < len(route.ids) else np.zeros(3)
        has_next = j + 1 < len(route.ids)
        vnext = vout[j + 1] if has_next else np.zeros(3)

        def fun(epoch):
            return insertion_value(
                epoch, k, t0, t1, pos[j], r1, vin[j], vnext, has_next, states, maxrev
            )[0]

        opt = minimize_scalar(
            fun,
            bounds=(max(t0 + 181, t - step), min(upper - 181, t + step)),
            method="bounded",
            options={"xatol": 0.05},
        )
        t = float(opt.x) if opt.fun < fun(t) else t
        _, b1, b2 = insertion_value(
            t, k, t0, t1, pos[j], r1, vin[j], vnext, has_next, states, maxrev
        )
        ids = route.ids.copy()
        times = route.times.copy()
        branches = route.branches.copy()
        ids.insert(j, k)
        times.insert(j, t)
        if j < len(branches):
            branches[j] = BRANCHES[b2].tolist()
        branches.insert(j, BRANCHES[b1].tolist())
        candidates.append(evaluated(ids, times, branches, route.initial, states))
    candidates.sort(key=lambda r: r.cost)
    if rng is not None and len(candidates) > 1 and (rng.random() < 0.3):
        weights = np.exp(
            -np.minimum(100, (np.array([r.cost for r in candidates]) - candidates[0].cost) / 0.65)
        )
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
    return candidates[0]
