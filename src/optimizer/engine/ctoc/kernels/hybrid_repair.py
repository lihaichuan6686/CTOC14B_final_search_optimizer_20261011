"""Destroy/repair flyby constraints while retaining movable impulse nodes."""

import copy
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from .physics import lambert, target_state
from .advanced_search import insertion_grid, insertion_value, BRANCHES
from .free_impulse import node_local


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
def replacement_grid(k, states, ids, r, t, vin, vout, branches):
    n = len(t) - 1
    records = np.full((n, 3), 1000000000.0)
    for j in range(1, n):
        if ids[j] >= 0:
            continue
        base = node_local(j, r[j], t[j], r, t, vout, vin[1:], branches, vin[0], 0.0)
        count = max(4, int((t[j + 1] - t[j - 1]) / 700))
        for it in range(count + 1):
            tt = (
                t[j]
                if it == count
                else t[j - 1] + 181 + (t[j + 1] - t[j - 1] - 362) * it / (count - 1)
            )
            rr = target_state(states, k, tt)[0]
            value = node_local(j, rr, tt, r, t, vout, vin[1:], branches, vin[0], 0.0) - base
            if value < records[j, 0]:
                records[j] = np.array([value, j, tt])
    return records


def replace_free(data, k, states, rng):
    r, vin, vout = cache(data)
    t = np.array(data["times"])
    ids = np.array(data["node_ids"])
    b = np.array(data["branches"])
    records = replacement_grid(k, states, ids, r, t, vin, vout, b)
    candidates = []
    for row in sorted(records, key=lambda v: v[0])[:5]:
        _, j, tt = row
        j = int(j)
        if j <= 0 or j >= len(t) - 1:
            continue

        def f(tm):
            return node_local(
                j, target_state(states, k, tm)[0], tm, r, t, vout, vin[1:], b, vin[0], 0.0
            )

        fit = minimize_scalar(
            f,
            bounds=(max(t[j - 1] + 181, tt - 700), min(t[j + 1] - 181, tt + 700)),
            method="bounded",
            options={"xatol": 0.05},
        )
        if fit.fun < f(tt):
            tt = float(fit.x)
        d = copy.deepcopy(data)
        d["node_ids"][j] = int(k)
        d["times"][j] = float(tt)
        d["positions"][j] = target_state(states, k, tt)[0].tolist()
        d["cost"] = actual_cost(d)
        candidates.append(d)
    if not candidates:
        return None
    return min(candidates, key=lambda d: d["cost"])


def insert(data, k, states, rng, explore=False):
    r, vin, vout = cache(data)
    times = np.array(data["times"][1:])
    ids = np.array(data["node_ids"][1:])
    rec = insertion_grid(ids, times, r, vin, vout, k, states, 700.0, 1)
    candidates = []
    for row in sorted(rec, key=lambda v: v[0])[:5]:
        _, j, t, _, _ = row
        j = int(j)
        t0 = data["times"][j]
        t1 = data["times"][j + 1] if j < len(times) else -1
        upper = t1 if t1 >= 0 else 864000.0
        r1 = r[j + 1] if j < len(times) else np.zeros(3)
        hasn = j + 1 < len(times)
        vnext = vout[j + 1] if hasn else np.zeros(3)
        if upper - t0 < 363:
            continue
        f = lambda tt: insertion_value(tt, k, t0, t1, r[j], r1, vin[j], vnext, hasn, states, 1)[0]
        fit = minimize_scalar(
            f,
            bounds=(max(t0 + 181, t - 700), min(upper - 181, t + 700)),
            method="bounded",
            options={"xatol": 0.1},
        )
        t = float(fit.x) if fit.fun < f(t) else float(t)
        _, b1, b2 = insertion_value(t, k, t0, t1, r[j], r1, vin[j], vnext, hasn, states, 1)
        from .physics import target_state

        d = copy.deepcopy(data)
        d["node_ids"].insert(j + 1, int(k))
        d["times"].insert(j + 1, t)
        d["positions"].insert(j + 1, target_state(states, k, t)[0].tolist())
        if j < len(d["branches"]):
            d["branches"][j] = BRANCHES[b2].tolist()
        d["branches"].insert(j, BRANCHES[b1].tolist())
        d["cost"] = actual_cost(d)
        candidates.append(d)
    candidates.sort(key=lambda d: d["cost"])
    replacement = replace_free(data, k, states, rng)
    if replacement is not None:
        candidates.append(replacement)
        candidates.sort(key=lambda d: d["cost"])
    if not candidates:
        raise ValueError("No insertion candidate")
    if explore:
        eligible = [d for d in candidates if d["cost"] < candidates[0]["cost"] + 1.2]
        weights = np.exp(-np.array([d["cost"] - eligible[0]["cost"] for d in eligible]) / 0.35)
        return eligible[int(rng.choice(len(eligible), p=weights / weights.sum()))]
    if (
        rng.random() < 0.25
        and len(candidates) > 1
        and (candidates[1]["cost"] < candidates[0]["cost"] + 0.25)
    ):
        return candidates[1]
    return candidates[0]
