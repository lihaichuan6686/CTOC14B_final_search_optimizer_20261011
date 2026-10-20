"""中途脉冲重搜的数值内核：直接转移与 coast+Lambert 两种候选共同竞争。

direct 每行8列：目标局部编号、网格编号、到达秒、分支编号、Δv、到达速度(3)。
midcourse 在上述8列后加：中途秒、位置(3)、滑行初速度(3)、滑行时间比例。
batch 对束中各父链并行，保留 Numba 的计算性能。
"""

import numpy as np
from numba import njit, prange
from orbit_dynamics import MU, RE, lambert, arc_min_radius, target_state


@njit(cache=True)
def schedule_score(tm, remaining, tleft, tright, penalty, pressure_hours=0.0, pressure_weight=0.0):
    pressure = remaining * pressure_hours * 3600 / max(tright - tm, 180.0)
    return (
        penalty * (tm - tleft) / (tright - tleft) + pressure_weight * max(0.0, pressure - 0.7) ** 2
    )


@njit(cache=True)
def expand(
    r,
    v,
    t,
    mask,
    depth,
    times,
    positions,
    counts,
    branches,
    tleft,
    tright,
    nkeep,
    penalty,
    pressure_hours=0.0,
    pressure_weight=0.0,
    max_transfer_s=0.0,
):
    n = len(counts)
    out = np.full((n * nkeep, 8), np.nan)
    row = 0
    for k in range(n):
        if mask & 1 << k:
            continue
        best = np.full(nkeep, 1000000000000.0)
        data = np.zeros((nkeep, 8))
        for a in range(counts[k]):
            tm = times[k, a]
            if tm - t < 180 or tright - tm < (n - depth) * 180:
                continue
            if max_transfer_s > 0 and tm - t > max_transfer_s:
                continue
            for b in range(len(branches)):
                va, vz = lambert(
                    r, positions[k, a], tm - t, branches[b, 0], branches[b, 1], branches[b, 2]
                )
                if not np.isfinite(va[0]):
                    continue
                dv = np.linalg.norm(va - v)
                rank = dv + schedule_score(
                    tm, n - depth - 1, tleft, tright, penalty, pressure_hours, pressure_weight
                )
                worst = np.argmax(best)
                if rank >= best[worst]:
                    continue
                if arc_min_radius(r, va, positions[k, a], vz, tm - t) < RE + 203:
                    continue
                duplicate = -1
                for q in range(nkeep):
                    if (
                        best[q] < 100000000000.0
                        and abs(data[q, 2] - tm) < 600
                        and (int(data[q, 3]) == b)
                    ):
                        duplicate = q
                if duplicate >= 0:
                    if best[duplicate] <= rank:
                        continue
                    worst = duplicate
                best[worst] = rank
                data[worst] = np.array([float(k), float(a), tm, float(b), dv, vz[0], vz[1], vz[2]])
        for q in range(nkeep):
            if best[q] < 100000000000.0:
                out[row] = data[q]
                row += 1
    return out[:row]


@njit(cache=True)
def elliptic_coast(r, v, dt):
    rn = np.linalg.norm(r)
    vv = np.dot(v, v)
    a = 1 / (2 / rn - vv / MU)
    if a <= 0:
        return (np.full(3, np.nan), np.full(3, np.nan), -1.0)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn < 1e-10:
        return (np.full(3, np.nan), np.full(3, np.nan), -1.0)
    ev = ((vv - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(ev)
    if e >= 1:
        return (np.full(3, np.nan), np.full(3, np.nan), -1.0)
    if e < 1e-09:
        p = r / rn
        E0 = 0.0
    else:
        p = ev / e
        E0 = np.arctan2(np.dot(r, v) / (e * np.sqrt(MU * a)), (1 - rn / a) / e)
    q = np.cross(h / hn, p)
    M = (E0 - e * np.sin(E0) + np.sqrt(MU / a**3) * dt + np.pi) % (2 * np.pi) - np.pi
    E = M
    for _ in range(35):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta
        if abs(delta) < 1e-13:
            break
    root = np.sqrt(1 - e * e)
    pos = a * ((np.cos(E) - e) * p + root * np.sin(E) * q)
    vel = np.sqrt(MU * a) / np.linalg.norm(pos) * (-np.sin(E) * p + root * np.cos(E) * q)
    return (pos, vel, a * (1 - e))


@njit(cache=True)
def dsm_expand(
    r,
    v,
    t,
    mask,
    depth,
    times,
    positions,
    counts,
    branches,
    tleft,
    tright,
    direct,
    penalty,
    blended=False,
    pressure_hours=0.0,
    pressure_weight=0.0,
    max_transfer_s=0.0,
    free_midcourse=False,
):
    n = len(counts)
    out = np.full((n * 3, 16), np.nan)
    row = 0
    for k in range(n):
        if mask & 1 << k:
            continue
        options = np.zeros(counts[k], np.bool_)
        for q in direct:
            if int(q[0]) == k:
                options[int(q[1])] = True
        close = np.full(2, 1000000000000.0)
        indices = np.full(2, -1, np.int64)
        for a in range(counts[k]):
            tm = times[k, a]
            if tm - t < 600 or tright - tm < (n - depth) * 180:
                continue
            if max_transfer_s > 0 and tm - t > max_transfer_s:
                continue
            pos, vel, rad = elliptic_coast(r, v, tm - t)
            if not np.isfinite(pos[0]):
                continue
            value = np.linalg.norm(pos - positions[k, a]) / (tm - t)
            w = np.argmax(close)
            if value < close[w]:
                close[w] = value
                indices[w] = a
        for a in indices:
            if a >= 0:
                options[a] = True
        best = np.full(3, 1000000000000.0)
        data = np.zeros((3, 16))
        for a in range(counts[k]):
            if not options[a]:
                continue
            tm = times[k, a]
            duration = tm - t
            if duration < 600 or tright - tm < (n - depth) * 180:
                continue
            if max_transfer_s > 0 and duration > max_transfer_s:
                continue
            radial = r / np.linalg.norm(r)
            tangent = v - np.dot(v, radial) * radial
            reference = np.full(3, np.nan)
            reference_cost = 1000000000000.0
            if blended:
                for branch in branches:
                    aa, zz = lambert(r, positions[k, a], duration, branch[0], branch[1], branch[2])
                    if np.isfinite(aa[0]):
                        c = np.linalg.norm(aa - v)
                        if c < reference_cost:
                            reference_cost = c
                            reference = aa
            for mode in range(5 if blended else 3):
                if mode < 3:
                    launch = v + (mode - 1) * 0.03 * tangent
                elif np.isfinite(reference[0]):
                    launch = v + (0.5 if mode == 3 else 1.25) * (reference - v)
                else:
                    continue
                burn = np.linalg.norm(launch - v)
                fractions = (
                    np.array([0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9])
                    if free_midcourse
                    else np.array([0.35, 0.65])
                )
                for fraction in fractions:
                    mid = t + fraction * duration
                    rm, vm, rp = elliptic_coast(r, launch, mid - t)
                    if not np.isfinite(rm[0]):
                        continue
                    if rp < RE + 203 and arc_min_radius(r, launch, rm, vm, mid - t) < RE + 203:
                        continue
                    for b in range(len(branches)):
                        aa, zz = lambert(
                            rm,
                            positions[k, a],
                            tm - mid,
                            branches[b, 0],
                            branches[b, 1],
                            branches[b, 2],
                        )
                        if not np.isfinite(aa[0]):
                            continue
                        cost = burn + np.linalg.norm(aa - vm)
                        rank = cost + schedule_score(
                            tm,
                            n - depth - 1,
                            tleft,
                            tright,
                            penalty,
                            pressure_hours,
                            pressure_weight,
                        )
                        w = np.argmax(best)
                        if rank >= best[w]:
                            continue
                        if arc_min_radius(rm, aa, positions[k, a], zz, tm - mid) < RE + 203:
                            continue
                        duplicate = -1
                        for j in range(3):
                            if (
                                best[j] < 100000000000.0
                                and abs(data[j, 2] - tm) < 600
                                and (int(data[j, 3]) == b)
                            ):
                                duplicate = j
                        if duplicate >= 0:
                            if best[duplicate] <= rank:
                                continue
                            w = duplicate
                        best[w] = rank
                        data[w] = np.array(
                            [
                                float(k),
                                float(a),
                                tm,
                                float(b),
                                cost,
                                zz[0],
                                zz[1],
                                zz[2],
                                mid,
                                rm[0],
                                rm[1],
                                rm[2],
                                launch[0],
                                launch[1],
                                launch[2],
                                fraction,
                            ]
                        )
        for j in range(3):
            if best[j] < 100000000000.0:
                out[row] = data[j]
                row += 1
    return out[:row]


@njit(cache=True)
def trial(
    tm,
    row,
    r,
    v,
    t,
    states,
    ids,
    branches,
    nremaining,
    tleft,
    tright,
    penalty,
    hours,
    weight,
    normal,
):
    k = int(row[0])
    branch = branches[int(row[3])]
    pos = target_state(states, ids[k], tm)[0]
    result = row.copy()
    result[2] = tm
    if abs(np.dot(pos, normal)) / np.linalg.norm(pos) > np.sin(np.pi / 12):
        return (1000000000000.0, result)
    if len(row) == 8:
        va, vz = lambert(r, pos, tm - t, branch[0], branch[1], branch[2])
        if not np.isfinite(va[0]) or arc_min_radius(r, va, pos, vz, tm - t) < RE + 203:
            return (1000000000000.0, result)
        dv = np.linalg.norm(va - v)
    else:
        middle = t + row[15] * (tm - t)
        rm, vm, rad = elliptic_coast(r, row[12:15], middle - t)
        if not np.isfinite(rm[0]):
            return (1000000000000.0, result)
        if rad < RE + 203 and arc_min_radius(r, row[12:15], rm, vm, middle - t) < RE + 203:
            return (1000000000000.0, result)
        va, vz = lambert(rm, pos, tm - middle, branch[0], branch[1], branch[2])
        if not np.isfinite(va[0]) or arc_min_radius(rm, va, pos, vz, tm - middle) < RE + 203:
            return (1000000000000.0, result)
        dv = np.linalg.norm(row[12:15] - v) + np.linalg.norm(va - vm)
        result[8] = middle
        result[9:12] = rm
    result[4] = dv
    result[5:8] = vz
    return (dv + schedule_score(tm, nremaining, tleft, tright, penalty, hours, weight), result)


@njit(cache=True)
def fraction_polish(
    row, r, v, t, states, ids, branches, nremaining, tleft, tright, penalty, hours, weight, normal
):
    duration = row[2] - t
    lower = max(0.01, 30.0 / duration)
    upper = min(0.99, 1.0 - 30.0 / duration)
    grid = np.linspace(lower, upper, 11)
    values = np.full(11, 1000000000000.0)
    best = row[4] + schedule_score(row[2], nremaining, tleft, tright, penalty, hours, weight)
    chosen = row.copy()
    for j in range(11):
        probe = row.copy()
        probe[15] = grid[j]
        value, q = trial(
            row[2],
            probe,
            r,
            v,
            t,
            states,
            ids,
            branches,
            nremaining,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
        )
        values[j] = value
        if value < best:
            best = value
            chosen = q.copy()
    j = int(np.argmin(values))
    lo = grid[max(0, j - 1)]
    hi = grid[min(10, j + 1)]
    x = hi - 0.61803398875 * (hi - lo)
    y = lo + 0.61803398875 * (hi - lo)
    for _ in range(14):
        probe = row.copy()
        probe[15] = x
        cx, qx = trial(
            row[2],
            probe,
            r,
            v,
            t,
            states,
            ids,
            branches,
            nremaining,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
        )
        probe[15] = y
        cy, qy = trial(
            row[2],
            probe,
            r,
            v,
            t,
            states,
            ids,
            branches,
            nremaining,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
        )
        if cx < best:
            best = cx
            chosen = qx.copy()
        if cy < best:
            best = cy
            chosen = qy.copy()
        if cx < cy:
            hi = y
        else:
            lo = x
        x = hi - 0.61803398875 * (hi - lo)
        y = lo + 0.61803398875 * (hi - lo)
    return chosen


@njit(cache=True)
def polish(
    rows,
    r,
    v,
    t,
    states,
    ids,
    branches,
    nremaining,
    tleft,
    tright,
    penalty,
    hours,
    weight,
    normal,
    step,
    max_transfer_s=0.0,
    free_midcourse=False,
):
    out = rows.copy()
    for i in range(len(rows)):
        row = rows[i]
        best = row[4] + schedule_score(row[2], nremaining, tleft, tright, penalty, hours, weight)
        chosen = row.copy()
        lo = max(t + (180 if len(row) == 8 else 600), row[2] - step / 2)
        hi = min(tright - (nremaining + 1) * 180, row[2] + step / 2)
        if max_transfer_s > 0:
            hi = min(hi, t + max_transfer_s)
        if hi - lo < 1:
            continue
        x = hi - 0.61803398875 * (hi - lo)
        y = lo + 0.61803398875 * (hi - lo)
        cx, qx = trial(
            x,
            row,
            r,
            v,
            t,
            states,
            ids,
            branches,
            nremaining,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
        )
        cy, qy = trial(
            y,
            row,
            r,
            v,
            t,
            states,
            ids,
            branches,
            nremaining,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
        )
        for _ in range(14):
            if cx < best:
                best = cx
                chosen = qx.copy()
            if cy < best:
                best = cy
                chosen = qy.copy()
            if cx < cy:
                hi = y
                y = x
                cy = cx
                qy = qx.copy()
                x = hi - 0.61803398875 * (hi - lo)
                cx, qx = trial(
                    x,
                    row,
                    r,
                    v,
                    t,
                    states,
                    ids,
                    branches,
                    nremaining,
                    tleft,
                    tright,
                    penalty,
                    hours,
                    weight,
                    normal,
                )
            else:
                lo = x
                x = y
                cx = cy
                qx = qy.copy()
                y = lo + 0.61803398875 * (hi - lo)
                cy, qy = trial(
                    y,
                    row,
                    r,
                    v,
                    t,
                    states,
                    ids,
                    branches,
                    nremaining,
                    tleft,
                    tright,
                    penalty,
                    hours,
                    weight,
                    normal,
                )
        if cx < best:
            best = cx
            chosen = qx.copy()
        if cy < best:
            chosen = qy.copy()
        if free_midcourse and len(row) > 8:
            chosen = fraction_polish(
                chosen,
                r,
                v,
                t,
                states,
                ids,
                branches,
                nremaining,
                tleft,
                tright,
                penalty,
                hours,
                weight,
                normal,
            )
        out[i] = chosen
    return out


@njit(cache=True, parallel=True)
def batch(
    r,
    v,
    t,
    masks,
    depth,
    times,
    positions,
    counts,
    branches,
    tleft,
    tright,
    penalty,
    blended,
    hours,
    weight,
    states,
    ids,
    normal,
    step,
    max_transfer_s=0.0,
    free_midcourse=False,
):
    parents = len(t)
    n = len(counts)
    direct = np.full((parents, n * 4, 8), np.nan)
    staged = np.full((parents, n * 3, 16), np.nan)
    dc = np.zeros(parents, np.int64)
    sc = np.zeros(parents, np.int64)
    for i in prange(parents):
        a = expand(
            r[i],
            v[i],
            t[i],
            masks[i],
            depth,
            times,
            positions,
            counts,
            branches,
            tleft,
            tright,
            4,
            penalty,
            hours,
            weight,
            max_transfer_s,
        )
        b = dsm_expand(
            r[i],
            v[i],
            t[i],
            masks[i],
            depth,
            times,
            positions,
            counts,
            branches,
            tleft,
            tright,
            a,
            penalty,
            blended,
            hours,
            weight,
            max_transfer_s,
            free_midcourse,
        )
        a = polish(
            a,
            r[i],
            v[i],
            t[i],
            states,
            ids,
            branches,
            n - depth - 1,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
            step,
            max_transfer_s,
        )
        b = polish(
            b,
            r[i],
            v[i],
            t[i],
            states,
            ids,
            branches,
            n - depth - 1,
            tleft,
            tright,
            penalty,
            hours,
            weight,
            normal,
            step,
            max_transfer_s,
            free_midcourse,
        )
        dc[i] = len(a)
        sc[i] = len(b)
        direct[i, : len(a)] = a
        staged[i, : len(b)] = b
    return (direct, staged, dc, sc)


@njit(cache=True)
def first_seeds(times, positions, counts, normal, end, branches, keep, time_penalty=0.5):
    n = len(counts)
    out = np.full((n * keep, 14), np.nan)
    cursor = 0
    for k in range(n):
        best = np.full(keep, 1000000000000.0)
        rows = np.zeros((keep, 14))
        for a in range(counts[k]):
            tm = times[k, a]
            if tm < 900.0 or tm > min(64800.0, end * 0.65):
                continue
            target = positions[k, a]
            radial = target / np.linalg.norm(target)
            h = normal - np.dot(normal, radial) * radial
            h /= np.linalg.norm(h)
            tangent = np.cross(h, radial)
            for tilt in (-0.14, 0.0, 0.14):
                tau = np.cos(tilt) * tangent + np.sin(tilt) * h
                for j in range(48):
                    angle = 2 * np.pi * (j + 0.5) / 48
                    r = (RE + 605) * (np.cos(angle) * radial - np.sin(angle) * tau)
                    v = np.sqrt(MU / (RE + 605)) * (np.sin(angle) * radial + np.cos(angle) * tau)
                    for b in range(len(branches)):
                        va, vz = lambert(
                            r, target, tm, branches[b, 0], branches[b, 1], branches[b, 2]
                        )
                        if not np.isfinite(va[0]):
                            continue
                        dv = np.linalg.norm(va - v)
                        rank = dv + time_penalty * tm / end
                        w = np.argmax(best)
                        if rank >= best[w]:
                            continue
                        if arc_min_radius(r, va, target, vz, tm) < RE + 203:
                            continue
                        duplicate = -1
                        for z in range(keep):
                            if (
                                best[z] < 100000000000.0
                                and abs(rows[z, 2] - tm) < 1200
                                and (np.linalg.norm(rows[z, 5:8] - vz) < 0.1)
                            ):
                                duplicate = z
                                break
                        if duplicate >= 0:
                            if best[duplicate] <= rank:
                                continue
                            w = duplicate
                        best[w] = rank
                        rows[w] = np.array(
                            [
                                float(k),
                                float(a),
                                tm,
                                float(b),
                                dv,
                                vz[0],
                                vz[1],
                                vz[2],
                                r[0],
                                r[1],
                                r[2],
                                v[0],
                                v[1],
                                v[2],
                            ]
                        )
        for j in range(keep):
            if best[j] < 100000000000.0:
                out[cursor] = rows[j]
                cursor += 1
    return out[:cursor]


@njit(cache=True, parallel=True)
def next_cost(r, v, t, masks, times, positions, counts, tright):
    costs = np.zeros(len(t))
    for i in prange(len(t)):
        if masks[i] == (1 << len(counts)) - 1:
            continue
        best = 3.0
        for k in range(len(counts)):
            if masks[i] & 1 << k:
                continue
            lo = np.searchsorted(times[k, : counts[k]], t[i] + 300.0)
            hi = np.searchsorted(times[k, : counts[k]], min(tright, t[i] + 86400.0), side="right")
            for a in range(lo, hi, 2):
                dt = times[k, a] - t[i]
                for major in range(2):
                    va, vz = lambert(r[i], positions[k, a], dt, major, 0, 0)
                    if not np.isfinite(va[0]):
                        continue
                    value = np.linalg.norm(va - v[i])
                    if (
                        value < best
                        and arc_min_radius(r[i], va, positions[k, a], vz, dt) >= RE + 203
                    ):
                        best = value
        costs[i] = best
    return costs
