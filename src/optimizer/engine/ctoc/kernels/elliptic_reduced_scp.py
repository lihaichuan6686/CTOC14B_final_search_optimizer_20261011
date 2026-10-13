"""SCP in target-relative encounter coordinates, eliminating moving constraints.

An encounter is parameterized exactly as r_target(t)+offset, |offset|<=995 m.
The full chain rule adds target velocity to the STM time derivatives. This
avoids projecting a linearly moved encounter sphere after every SOCP step.
"""

import copy, json, time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from .physics import RE, target_state
from .elliptic_joint_scp import pack_initial, initial_elliptic, pulse_model
from .multi_impulse_round8 import ensure_offsets


def pack(d, states):
    n = len(d["branches"])
    r = np.array(d["positions"])[1:] / 30000
    t = np.array(d["times"])[1:] / 3600
    for j, k in enumerate(d["node_ids"][1:]):
        if k >= 0:
            r[j] = np.array(d["positions"][j + 1]) - target_state(states, k, t[j] * 3600)[0]
    return np.r_[pack_initial(d["initial"]), r.ravel(), t]


def evaluate(z, ids, branches, states, jac=True):
    n = len(branches)
    off = 6 + 3 * n
    x = z.copy()
    velocities = {}
    for j, k in enumerate(ids[1:]):
        if k >= 0:
            r, v = target_state(states, k, z[off + j] * 3600)
            x[6 + 3 * j : 9 + 3 * j] = (r + z[6 + 3 * j : 9 + 3 * j]) / 30000
            velocities[j] = v
    dv, J, rad = pulse_model(x, branches, jac)
    if jac:
        for j, v in velocities.items():
            columns = J[:, 6 + 3 * j : 9 + 3 * j].copy() / 30000
            J[:, off + j] += columns @ v * 3600
            J[:, 6 + 3 * j : 9 + 3 * j] = columns
    return (dv, J, rad, x)


def refine(data, states, minutes, output=None, weights=None, max_iterations=1500, encounter_radius=0.995, terminal_time_limit_hours=240.):
    if not np.isfinite(encounter_radius) or not 0 < encounter_radius <= 10000:
        raise ValueError("Invalid continuation radius")
    d = copy.deepcopy(data)
    ids = np.array(d["node_ids"])
    branches = np.array(d["branches"])
    n = len(branches)
    off = 6 + 3 * n
    visits = np.flatnonzero(ids[1:] >= 0)
    z = pack(d, states)
    scales = np.r_[[0.04, 0.04, 0.04, 0.03, 0.05, 0.05], np.full(3 * n, 0.015), np.full(n, 0.08)]
    for j in visits:
        scales[6 + 3 * j : 9 + 3 * j] = 0.25
    weights = np.ones(n) if weights is None else np.array(weights)
    if weights.shape != (n,) or np.any(weights <= 0) or (not np.all(np.isfinite(weights))):
        raise ValueError("Bad weights")
    radius = 0.3
    deadline = time.monotonic() + minutes * 60
    history = []
    stalls = 0
    for iteration in range(max_iterations):
        if time.monotonic() > deadline:
            break
        dv, J, _, _ = evaluate(z, ids, branches, states)
        old = float(weights @ np.linalg.norm(dv, axis=1))
        step = cp.Variable(len(z))
        bound = cp.Variable(n)
        dx = cp.multiply(scales, step)
        zz = z + dx
        impulses = cp.reshape(dv.ravel() + sp.csc_matrix(J) @ dx, (n, 3), order="C")
        rr = cp.reshape(zz[6:off], (n, 3), order="C")
        tt = zz[off:]
        cons = [
            cp.norm(impulses, axis=1) <= bound,
            cp.abs(step) <= radius,
            zz[1] >= -1.5707,
            zz[1] <= 1.5707,
            zz[3] >= -0.999999,
            zz[3] <= 0.999999,
            cp.norm(zz[4:6]) <= 0.9985,
            tt[0] >= d.get("minimum_gap_hours", 10.0/3600),
            tt[-1] <= min(240., float(terminal_time_limit_hours)),
            tt[1:] - tt[:-1] >= d.get("minimum_gap_hours", 10.0/3600),
            cp.norm(rr[visits], axis=1) <= encounter_radius,
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
            trial = z + np.array(step.value) * scales * fraction
            if np.linalg.norm(trial[4:6]) > 0.99850001:
                continue
            try:
                p, _, rad, _ = evaluate(trial, ids, branches, states, False)
                value = float(weights @ np.linalg.norm(p, axis=1))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                continue
            if rad >= RE + 202 and value < old - 1e-10:
                z = trial
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
    dv, _, rad, x = evaluate(z, ids, branches, states, False)
    init = initial_elliptic(x)
    d.update(
        initial=init.tolist(),
        positions=np.vstack([init[:3], x[6:off].reshape(n, 3) * 30000]).tolist(),
        times=np.r_[0, x[off:] * 3600].tolist(),
        cost=float(np.linalg.norm(dv, axis=1).sum()),
        reduced_scp_history=history,
        minimum_arc_radius_km=float(rad),
    )
    ensure_offsets(d, states)
    if output is not None:
        from .basin_escape import save

        save(d, output)
    print("REDUCED_RESULT", d["cost"], flush=True)
    return d
