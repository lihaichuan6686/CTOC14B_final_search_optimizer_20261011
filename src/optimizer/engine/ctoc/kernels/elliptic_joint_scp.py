"""Jointly vary all six initial elements, maneuver nodes, and flyby epochs.

Nonsingular eccentricity components e*cos(f), e*sin(f) have an exact SOC
bound. Positions, velocities, and ephemerides use the calibrated pole frame
during optimization and are all rotated back to ICRF before export.
"""

import copy, json, math, time
import numpy as np
from numba import njit
import scipy.sparse as sp
import cvxpy as cp
from .physics import MU, RE, target_state
from .j2_refine import shoot, acceleration_hessian
from .basin_escape import save


@njit(cache=True)
def initial_elliptic(x):
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
    n = len(branches)
    off = 6 + 3 * n
    dim = len(x)
    init = initial_elliptic(x)
    r = np.empty((n + 1, 3))
    r[0] = init[:3]
    r[1:] = x[6:off].reshape(n, 3) * 30000
    t = np.zeros(n + 1)
    t[1:] = x[off:] * 3600
    dv = np.zeros((n, 3))
    jac = np.zeros((n * 3, dim))
    previous = init[3:]
    minimum = 1000000000.0
    dr = np.zeros((n + 1, 3, dim))
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
        for j in range(n):
            for axis in range(3):
                dr[j + 1, axis, 6 + 3 * j + axis] = 30000
    for i in range(n):
        vi, vf, phi, rad, error = shoot(r[i], r[i + 1], t[i + 1] - t[i], branches[i])
        if error > 0.001 or not np.isfinite(vi[0]):
            raise ValueError("Elliptic J2 shooting failed")
        dv[i] = vi - previous
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
            startjac[:, off + i] += 3600 * vt
            endjac[:, off + i] += 3600 * wt
            if i:
                startjac[:, off + i - 1] -= 3600 * vt
                endjac[:, off + i - 1] -= 3600 * wt
            jac[3 * i : 3 * i + 3] = startjac - previous_jac
            previous_jac = endjac
    return (dv, jac, minimum)


def optimize(
    data,
    states,
    minutes,
    output,
    ball_radius=0.995,
    *,
    pulse_weights=None,
    initial_radius=0.3,
    max_iterations=600,
    step_scale=1.0,
    gain_tolerance=1e-09,
):
    data = copy.deepcopy(data)
    ids = np.array(data["node_ids"])
    n = len(ids) - 1
    branches = np.array(data["branches"])
    off = 6 + 3 * n
    visits = np.flatnonzero(ids >= 0)
    target_ids = ids[visits]
    x = np.r_[
        pack_initial(data["initial"]),
        np.array(data["positions"])[1:].ravel() / 30000,
        np.array(data["times"])[1:] / 3600,
    ]
    weights = np.ones(n) if pulse_weights is None else np.asarray(pulse_weights, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Pulse weights must be finite, positive, and match the arc count")
    if step_scale <= 0 or initial_radius <= 0:
        raise ValueError("Trust-region scales must be positive")
    scales = (
        np.r_[[0.02, 0.02, 0.02, 0.02, 0.03, 0.03], np.full(3 * n, 0.001), np.full(n, 0.02)]
        * step_scale
    )
    deadline = time.monotonic() + 60 * minutes
    radius = initial_radius
    history = []
    stalled = 0
    for iteration in range(max_iterations):
        if time.monotonic() > deadline:
            break
        dv, J, mr = pulse_model(x, branches)
        old = float(weights @ np.linalg.norm(dv, axis=1))
        centers = []
        velocities = []
        for k, j in zip(target_ids, visits):
            r, v = target_state(states, k, x[off + j - 1] * 3600)
            centers.append(r)
            velocities.append(v)
        centers = np.array(centers)
        velocities = np.array(velocities)
        delta = cp.Variable(len(x))
        bound = cp.Variable(n)
        dx = cp.multiply(scales, delta)
        xx = x + dx
        impulses = cp.reshape(dv.ravel() + sp.csc_matrix(J) @ dx, (n, 3), order="C")
        rr = cp.reshape(xx[6:off], (n, 3), order="C") * 30000
        tt = xx[off:]
        moving = centers + cp.multiply(
            velocities,
            cp.reshape((tt[visits - 1] - x[off:][visits - 1]) * 3600, (len(visits), 1), order="C"),
        )
        constraints = [
            cp.norm(impulses, axis=1) <= bound,
            cp.abs(delta) <= radius,
            xx[1] >= -1.5707,
            xx[1] <= 1.5707,
            xx[3] >= -0.999999,
            xx[3] <= 0.999999,
            cp.norm(xx[4:6]) <= 0.9985,
            tt[0] >= 0.05,
            tt[-1] <= 240,
            tt[1:] - tt[:-1] >= data.get("minimum_gap_hours", 0.005),
            cp.norm(rr[visits - 1] - moving, axis=1) <= ball_radius,
        ]
        problem = cp.Problem(
            cp.Minimize(weights @ bound + 1e-09 * cp.sum_squares(delta)), constraints
        )
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
        if delta.value is None:
            radius *= 0.4
            continue
        if (
            not np.all(np.isfinite(delta.value))
            or bound.value is None
            or (not np.all(np.isfinite(bound.value)))
        ):
            radius *= 0.4
            continue
        pred = old - float(weights @ bound.value)
        accepted = False
        value = old
        for fraction in [1.0, 0.5, 0.25, 0.125]:
            trial = x + np.asarray(delta.value) * scales * fraction
            if np.linalg.norm(trial[4:6]) > 0.99850001:
                continue
            rr = trial[6:off].reshape(n, 3) * 30000
            for k, j in zip(target_ids, visits):
                center = target_state(states, k, trial[off + j - 1] * 3600)[0]
                offset = rr[j - 1] - center
                length = np.linalg.norm(offset)
                if length > ball_radius:
                    rr[j - 1] = center + offset * ball_radius / length
            trial[6:off] = rr.ravel() / 30000
            try:
                pv, _, rad = pulse_model(trial, branches, False)
                cost = float(weights @ np.linalg.norm(pv, axis=1))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                continue
            if rad >= RE + 202 and cost < old - gain_tolerance:
                x = trial
                value = cost
                accepted = True
                break
        if accepted:
            stalled = 0
            ratio = (old - value) / max(pred, 1e-15)
            if ratio > 0.6:
                radius = min(8.0, 1.5 * radius)
            elif ratio < 0.15:
                radius = max(0.0001, 0.6 * radius)
        else:
            stalled += 1
            radius *= 0.4
        true_cost = float(np.linalg.norm(pv if accepted else dv, axis=1).sum())
        record = dict(
            iteration=iteration,
            cost_km_s=true_cost,
            weighted_objective=value,
            predicted_gain_km_s=pred,
            radius=radius,
            accepted=accepted,
        )
        history.append(record)
        if iteration % 10 == 0 or not accepted:
            print("ELLIPTIC_SCP", json.dumps(record), flush=True)
        if (
            stalled >= 8
            or radius < 1e-06
            or (pred < 5 * gain_tolerance and old - value < 5 * gain_tolerance)
        ):
            break
    init = initial_elliptic(x)
    r = np.vstack([init[:3], x[6:off].reshape(n, 3) * 30000])
    t = np.r_[0.0, x[off:] * 3600]
    offsets = np.zeros_like(r)
    for k, j in zip(target_ids, visits):
        offsets[j] = r[j] - target_state(states, k, t[j])[0]
    pulses, _, rad = pulse_model(x, branches, False)
    data.update(
        initial=init.tolist(),
        positions=r.tolist(),
        times=t.tolist(),
        encounter_offsets_km=offsets.tolist(),
        cost=float(np.linalg.norm(pulses, axis=1).sum()),
        elliptic_joint_history=history,
        minimum_arc_radius_km=float(rad),
        pulse_objective_weights=weights.tolist(),
        weighted_objective=float(weights @ np.linalg.norm(pulses, axis=1)),
    )
    if output is not None:
        save(data, output)
    print("ELLIPTIC_RESULT", data["cost"], flush=True)
    return data
