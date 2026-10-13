"""Nonmonotone basin hopping with orbital-frame transport and mesh changes.

All costs in this search are two-body screens. Intermediate uphill moves are
allowed; only subsequent native and official verification permits publication.
"""

import copy
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .physics import MU, RE, norm, target_state, lambert, arc_min_radius
from .advanced_search import arc_choices, BRANCHES
from .hybrid_repair import actual_cost


def save(data, path):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for delay in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(delay)
    pending = path.with_name(path.stem + f".pending_{time.time_ns()}.json")
    temp.replace(pending)
    print("CHECKPOINT_PENDING", str(pending), flush=True)


def orbit_frame(r, v):
    radial = r / norm(r)
    normal = np.cross(r, v)
    normal /= norm(normal)
    return np.column_stack((radial, np.cross(normal, radial), normal))


def retime_lift(data, ids, times, states):
    """Transport free nodes between changed target orbital frames.

    Initial position and velocity rotate together, preserving a and e.
    Encounter order may change; node times interpolate within each visit leg.
    """
    d = copy.deepcopy(data)
    oldr = np.array(d["positions"])
    oldtime = np.array(d["times"])
    visits = np.flatnonzero(np.array(d["node_ids"]) >= 0)
    oldids = np.array(d["node_ids"])[visits]
    if len(ids) != len(visits):
        raise ValueError("Visit count differs")
    if np.min(np.diff(np.r_[0, times])) < 10 or times[-1] > 864000:
        raise ValueError("Invalid target epochs")
    rotations = []
    for j, node in enumerate(visits):
        ra, va = target_state(states, int(oldids[j]), oldtime[node])
        rb, vb = target_state(states, int(ids[j]), times[j])
        rotations.append(orbit_frame(rb, vb) @ orbit_frame(ra, va).T)
    knots = np.r_[0, visits]
    rotations = np.array([rotations[0], *rotations])
    transform = Slerp(oldtime[knots], Rotation.from_matrix(rotations))(oldtime)
    r = transform.apply(oldr)
    t = np.interp(np.arange(len(oldtime)), knots, np.r_[0, times])
    for j in range(len(visits)):
        left = 0 if j == 0 else visits[j - 1]
        right = visits[j]
        fraction = (oldtime[left : right + 1] - oldtime[left]) / (oldtime[right] - oldtime[left])
        t[left : right + 1] = (0 if j == 0 else times[j - 1]) + fraction * (
            times[j] - (0 if j == 0 else times[j - 1])
        )
    if np.min(np.diff(t)) < 10:
        raise ValueError("Transported free-node epochs too close")
    for j, node in enumerate(visits):
        d["node_ids"][node] = int(ids[j])
        r[node] = target_state(states, int(ids[j]), float(t[node]))[0]
    initial = np.r_[r[0], rotations[0] @ np.array(d["initial"][3:])]
    d.update(
        initial=initial.tolist(), positions=r.tolist(), times=t.tolist(), minimum_gap_hours=10.0/3600
    )
    d.pop("encounter_offsets_km", None)
    d.pop("encounter_ball_history", None)
    return d


def node_branch_dp(data, maxrev=2):
    d = copy.deepcopy(data)
    r = np.array(d["positions"])
    t = np.array(d["times"])
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
    d["initial"][3:] = circle[path[0]].tolist()
    d["branches"] = BRANCHES[path].tolist()
    d["cost"] = total
    return d


def compact(data, threshold=0.0005):
    """Remove a free node only if the recomputed chain changes by < threshold."""
    d = copy.deepcopy(data)
    old = actual_cost(d)
    i = 1
    while i < len(d["times"]) - 1:
        if d["node_ids"][i] >= 0:
            i += 1
            continue
        candidates = []
        for branch in BRANCHES:
            trial = copy.deepcopy(d)
            for key in ["positions", "times", "node_ids"]:
                trial[key].pop(i)
            trial["branches"][i - 1 : i + 1] = [branch.tolist()]
            a, z = lambert(
                np.array(trial["positions"][i - 1]),
                np.array(trial["positions"][i]),
                trial["times"][i] - trial["times"][i - 1],
                *branch,
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
            old, d = min(candidates, key=lambda p: p[0])
        else:
            i += 1
    d["cost"] = old
    return d


def perturb(data, states, rng, iteration):
    visits = np.flatnonzero(np.array(data["node_ids"]) >= 0)
    ids = np.array(data["node_ids"])[visits]
    tt = np.array(data["times"])[visits]
    original = tt.copy()
    mode = iteration % 5
    description = {}
    if mode in [0, 2]:
        count = 1 if mode == 0 else int(rng.integers(2, 5))
        first = int(rng.integers(0, len(ids) - count + 1))
        amount = float(rng.choice([0.5, 2, 6, 12, 24, 48]) * 3600 * rng.choice([-1, 1]))
        tt[first : first + count] = np.clip(tt[first : first + count] + amount, 10.0, 864000.0)
        order = np.argsort(tt)
        tt = tt[order]
        ids = ids[order]
        description = {
            "kind": "visit_epoch_relocation",
            "first": first,
            "count": count,
            "shift_h": amount / 3600,
        }
    elif mode == 1:
        duration = np.diff(np.r_[0, tt])
        noise = rng.normal(size=len(tt))
        width = int(rng.choice([2, 4, 7, 12]))
        noise = np.convolve(noise, np.ones(width) / np.sqrt(width), mode="same")
        strength = float(rng.choice([0.003, 0.008, 0.015, 0.04, 0.12]))
        weighted = np.maximum(10.0, duration * np.exp(strength * noise))
        finish = float(
            np.clip(tt[-1] + rng.normal(0, min(5, 12 * strength) * 3600), 180000, 864000)
        )
        tt = np.cumsum(10 + (finish - len(tt) * 10) * (weighted / weighted.sum()))
        description = {"kind": "wide_duration_warp", "strength": strength, "width": width}
    elif mode == 3:
        first = int(rng.integers(0, 33))
        length = int(rng.integers(1, 5))
        block = ids[first : first + length].copy()
        remainder = np.r_[ids[:first], ids[first + length :]]
        destination = int(
            np.clip(first + rng.choice([-5, -3, -2, -1, 1, 2, 3, 5]), 0, len(remainder))
        )
        ids = np.r_[remainder[:destination], block, remainder[destination:]]
        description = {
            "kind": "sequence_block_relocation",
            "from": first,
            "length": len(block),
            "to": destination,
        }
    else:
        i = int(rng.integers(len(ids)))
        j = int(np.clip(i + rng.choice([-7, -3, -2, -1, 1, 2, 3, 7]), 0, len(ids) - 1))
        if i == j:
            j = (i + 1) % len(ids)
        ids[i], ids[j] = (ids[j], ids[i])
        description = {"kind": "distant_target_swap", "indices": [int(i), int(j)]}
    description["maximum_target_shift_h"] = float(np.max(np.abs(tt - original)) / 3600)
    return (retime_lift(data, ids, tt, states), description)
