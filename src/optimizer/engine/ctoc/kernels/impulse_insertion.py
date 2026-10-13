"""Dense, globally coupled mid-coast impulse insertion and adaptive SOCP.

Solve the full linearized pulse-norm problem to obtain consistent subgradients
at zero as well as nonzero burns. Interpolate their primer along each J2 coast
using STMs. A primer norm above one is a local descent indicator, not a global
certificate. Insert zero-burn free nodes at its strongest peaks, jointly
reoptimize all nodes/epochs, and retain only a true nonlinear cost reduction.
"""

import copy
import numpy as np
from .adaptive_scp import trajectory, branch_near


def add_zero_nodes(data, peaks):
    d = copy.deepcopy(data)
    for candidate in sorted(peaks, key=lambda v: v["arc"], reverse=True):
        i = candidate["arc"]
        r = np.array(d["positions"])
        t = d["times"]
        tm = candidate["time_s"]
        rm = candidate["position"]
        b0, _, _ = branch_near(r[i], rm, tm - t[i], candidate["v1"])
        b1, _, _ = branch_near(rm, r[i + 1], t[i + 1] - tm, candidate["v2"])
        d["positions"].insert(i + 1, rm.tolist())
        d["times"].insert(i + 1, tm)
        d["node_ids"].insert(i + 1, -1)
        d["branches"][i : i + 1] = [b0, b1]
    d["minimum_gap_hours"] = 10.0 / 3600
    d["cost"] = trajectory(d)[4]
    return d
