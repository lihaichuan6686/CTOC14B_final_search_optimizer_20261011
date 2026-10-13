"""Bounded optimization experiments rooted in a supplied native trajectory."""

import copy
import numpy as np
from .j2_refine import shoot, propagate
from .adaptive_scp import branch_near, trajectory


def regularize(d, max_angle=110.0):
    """Repair ill-conditioned long arcs by shorter, strictly converged BVPs.

    A long arc's unconverged endpoint is used only to initialize a midpoint.
    It is not accepted as a valid coast: both new BVPs must independently
    converge and the resulting entire trajectory is reevaluated strictly.
    """
    d = copy.deepcopy(d)
    i, inserted, records = (0, 0, [])
    while i < len(d["branches"]):
        r = np.array(d["positions"])
        t = np.array(d["times"])
        b = np.array(d["branches"][i])
        angle = np.degrees(
            np.arccos(
                np.clip(r[i] @ r[i + 1] / np.linalg.norm(r[i]) / np.linalg.norm(r[i + 1]), -1, 1)
            )
        )
        angle = (360 - angle if b[0] else angle) + 360 * b[1]
        vi, _, _, _, err = shoot(r[i], r[i + 1], t[i + 1] - t[i], b)
        if angle <= max_angle and err < 0.0001:
            i += 1
            continue
        if not np.all(np.isfinite(vi)) or err > 0.02:
            raise ValueError(f"Long-arc seed residual too large: arc={i}, error_km={err}")
        dt = (t[i + 1] - t[i]) / 2
        if dt < 20:
            raise ValueError("Cannot split a short invalid coast")
        midpoint, _, _ = propagate(r[i], vi, dt)
        b0, _, _ = branch_near(r[i], midpoint[:3], dt, vi)
        b1, _, _ = branch_near(midpoint[:3], r[i + 1], dt, midpoint[3:])
        records.append(
            dict(arc=i, seed_endpoint_residual_m=float(err * 1000), angle_deg=float(angle))
        )
        d["positions"].insert(i + 1, midpoint[:3].tolist())
        d["times"].insert(i + 1, float(t[i] + dt))
        d["node_ids"].insert(i + 1, -1)
        d["branches"][i : i + 1] = [b0, b1]
        inserted += 1
        if inserted > 250:
            raise ValueError("Excessive mesh regularization")
    _, _, _, _, d["cost"], rad = trajectory(d)
    d["minimum_arc_radius_km"] = rad
    d["external_regularization"] = records
    print("EXTERNAL_REGULARIZE", inserted, d["cost"], flush=True)
    return d
