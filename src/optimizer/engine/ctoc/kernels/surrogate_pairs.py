"""Two-new-impulse surrogate-primer test on each J2 coast.

For new impulses at a<b and an existing end impulse d, impose identical
end position and post-burn velocity. With w at b, w_a=-Q w and delta_d=-D w.
The directional cost is |w|+|Qw|-unit(d).D w. Thus the surrogate condition is
max_|u|=1 ((D.T unit(d)).u-|Qu|)>1. This is a local test only.
"""

import copy
import numpy as np
from scipy.optimize import minimize
from .adaptive_scp import trajectory, branch_near
from .j2_refine import propagate


def surrogate_maximum(Q, b):
    singular = np.linalg.svd(Q)
    if np.linalg.norm(b) <= 1 + singular[1][-1]:
        return None

    def value(u):
        q = Q @ u
        norm = max(np.linalg.norm(q), 1e-15)
        return (float(-b @ u + norm), -b + Q.T @ q / norm)

    seeds = [b / np.linalg.norm(b), singular[2][-1], -singular[2][-1]]
    best = None
    for seed in seeds:
        fit = minimize(
            value,
            seed,
            jac=True,
            method="SLSQP",
            constraints=[{"type": "eq", "fun": lambda u: u @ u - 1, "jac": lambda u: 2 * u}],
            options={"maxiter": 45, "ftol": 1e-11},
        )
        u = fit.x / np.linalg.norm(fit.x)
        score = float(b @ u - np.linalg.norm(Q @ u))
        if best is None or score > best[0]:
            best = (score, u)
    return best


def scan(data, samples=13, both_ends=False):
    vi, vf, phis, pulses, cost, rad = trajectory(data)
    r = np.array(data["positions"])
    t = np.array(data["times"])
    records = []
    candidates = []
    skipped = 0
    for i in range(len(vi) if both_ends else len(vi) - 1):
        final = pulses[i + 1] if i + 1 < len(vi) else np.zeros(3)
        use_end = np.linalg.norm(final) >= 1e-05
        use_start = both_ends and np.linalg.norm(pulses[i]) >= 1e-05
        if not use_end and (not use_start):
            continue
        duration = t[i + 1] - t[i]
        fractions = [f for f in np.linspace(0.04, 0.96, samples) if min(f, 1 - f) * duration >= 20]
        states = []
        onward = []
        for f in fractions:
            state, partial, _ = propagate(r[i], vi[i], duration * f)
            states.append(state)
            onward.append(np.linalg.solve(partial.T, phis[i].T).T)
        for a in range(len(fractions)):
            for b in range(a + 1, len(fractions)):
                if (fractions[b] - fractions[a]) * duration < 20:
                    continue
                M1, M2 = (onward[a], onward[b])
                cases = []
                if use_end:
                    if np.linalg.cond(M1[:3, 3:]) > 10000000000.0:
                        skipped += 1
                    else:
                        Q = np.linalg.solve(M1[:3, 3:], M2[:3, 3:])
                        D = M2[3:, 3:] - M1[3:, 3:] @ Q
                        cases.append(("ending_burn", Q, D.T @ (final / np.linalg.norm(final))))
                if use_start:
                    coupled = np.column_stack([phis[i][:, 3:], M1[:, 3:]])
                    if np.linalg.cond(coupled) > 1000000000000.0:
                        skipped += 1
                    else:
                        maps = -np.linalg.solve(coupled, M2[:, 3:])
                        A, B = (maps[:3], maps[3:])
                        cases.append(
                            ("starting_burn", B, -A.T @ (pulses[i] / np.linalg.norm(pulses[i])))
                        )
                for kind, Q, vector in cases:
                    result = surrogate_maximum(Q, vector)
                    bound = float(np.linalg.norm(vector) - np.linalg.svd(Q, compute_uv=False)[-1])
                    record = {
                        "arc": i,
                        "case": kind,
                        "t1_s": float(t[i] + fractions[a] * duration),
                        "t2_s": float(t[i] + fractions[b] * duration),
                        "upper_bound": bound,
                        "surrogate_norm": None if result is None else result[0],
                    }
                    records.append(record)
                    if result is not None and result[0] > 1.0005:
                        candidates.append(
                            {**record, "states": [states[a], states[b]], "initial_velocity": vi[i]}
                        )
    candidates.sort(key=lambda x: x["surrogate_norm"], reverse=True)
    unique_pairs = len({(q["arc"], q["t1_s"], q["t2_s"]) for q in records})
    return (
        candidates,
        {
            "cost_km_s": cost,
            "sample_count": len(records),
            "unique_time_pairs": unique_pairs,
            "singular_pairs_skipped": skipped,
            "pairs": records,
            "descent_pair_count": len(candidates),
            "interpretation": "Local surrogate-primer directions only; exact nonlinear and ATK validation required.",
        },
    )


def insert_pair(data, candidate):
    d = copy.deepcopy(data)
    i = candidate["arc"]
    r = np.array(d["positions"])
    t = d["times"]
    states = candidate["states"]
    positions = [r[i], states[0][:3], states[1][:3], r[i + 1]]
    times = [t[i], candidate["t1_s"], candidate["t2_s"], t[i + 1]]
    velocities = [candidate["initial_velocity"], states[0][3:], states[1][3:]]
    branches = [
        branch_near(positions[j], positions[j + 1], times[j + 1] - times[j], velocities[j])[0]
        for j in range(3)
    ]
    d["positions"][i + 1 : i + 1] = [states[0][:3].tolist(), states[1][:3].tolist()]
    d["times"][i + 1 : i + 1] = times[1:3]
    d["node_ids"][i + 1 : i + 1] = [-1, -1]
    d["branches"][i : i + 1] = branches
    if "encounter_offsets_km" in d:
        d["encounter_offsets_km"][i + 1 : i + 1] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    d["minimum_gap_hours"] = 10.0 / 3600
    d["cost"] = trajectory(d)[4]
    return d
