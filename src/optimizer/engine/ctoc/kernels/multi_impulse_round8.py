"""Finite multi-impulse topology excursions followed by deep true-fuel polish.

The temporary weighted norms are a search device, never a reported fuel score.
All experiments retain 35 encounters and six initial variables. Dynamics and
ephemerides are rotated together into the calibrated fixed-pole J2 frame.
"""

import copy
import numpy as np
from .physics import RE, lambert, target_state
from .j2_refine import shoot, propagate
from .adaptive_scp import trajectory, branch_near


def insert_coast(data, arc, fraction):
    """Insert a dynamically continuous zero-burn node, preserving every visit."""
    d = copy.deepcopy(data)
    r = np.array(d["positions"])
    t = np.array(d["times"])
    vi, _, _, _, err = shoot(r[arc], r[arc + 1], t[arc + 1] - t[arc], np.array(d["branches"][arc]))
    if err > 0.0001:
        raise ValueError("Invalid coast to subdivide")
    duration = (t[arc + 1] - t[arc]) * fraction
    if min(duration, t[arc + 1] - t[arc] - duration) < 20:
        raise ValueError("Too short a new coast")
    state, _, _ = propagate(r[arc], vi, duration)
    b0, _, _ = branch_near(r[arc], state[:3], duration, vi)
    b1, _, _ = branch_near(state[:3], r[arc + 1], t[arc + 1] - t[arc] - duration, state[3:])
    d["positions"].insert(arc + 1, state[:3].tolist())
    d["times"].insert(arc + 1, float(t[arc] + duration))
    d["node_ids"].insert(arc + 1, -1)
    d["branches"][arc : arc + 1] = [b0, b1]
    d.pop("encounter_offsets_km", None)
    return d


def remove_burn(data, k):
    """Bridge across a free burn, allowing a real finite fuel increase."""
    if k <= 0 or k >= len(data["times"]) - 1 or data["node_ids"][k] >= 0:
        raise ValueError("Only an interior free node can be removed")
    vi, vf, _, _, _, _ = trajectory(data)
    r = np.array(data["positions"])
    t = np.array(data["times"])
    incoming = np.array(data["initial"][3:]) if k == 1 else vf[k - 2]
    outgoing = vi[k + 1] if k + 1 < len(vi) else None
    choices = []
    for rev in range(3):
        for major in range(2):
            for side in range(1 if rev == 0 else 2):
                b = np.array([major, rev, side])
                a, z = lambert(r[k - 1], r[k + 1], t[k + 1] - t[k - 1], *b)
                if not np.all(np.isfinite(a)):
                    continue
                screening = np.linalg.norm(a - incoming) + (
                    np.linalg.norm(outgoing - z) if outgoing is not None else 0
                )
                choices.append((screening, b))
    best = None
    for _, b in sorted(choices, key=lambda q: q[0])[:5]:
        try:
            a, z, _, rad, err = shoot(r[k - 1], r[k + 1], t[k + 1] - t[k - 1], b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if err > 0.0001 or rad < RE + 202 or (not np.all(np.isfinite(a))):
            continue
        value = np.linalg.norm(a - incoming) + (
            np.linalg.norm(outgoing - z) if outgoing is not None else 0
        )
        if best is None or value < best[0]:
            best = (value, b.tolist())
    if best is None:
        raise ValueError("No feasible J2 bridge after burn removal")
    d = copy.deepcopy(data)
    for key in ("positions", "times", "node_ids"):
        d[key].pop(k)
    d["branches"][k - 1 : k + 1] = [best[1]]
    d.pop("encounter_offsets_km", None)
    return d


def ensure_offsets(d, states):
    offsets = np.zeros_like(np.array(d["positions"]))
    for j, k in enumerate(d["node_ids"]):
        if k >= 0:
            offsets[j] = np.array(d["positions"][j]) - target_state(states, k, d["times"][j])[0]
    d["encounter_offsets_km"] = offsets.tolist()
    return d


def check_complete(d):
    if sorted((k for k in d["node_ids"] if k >= 0)) != list(range(35)):
        raise ValueError("Incomplete or repeated visits")
    t = np.array(d["times"])
    if t[0] != 0 or t[-1] > 864000.00001 or np.min(np.diff(t)) < 9.999:
        raise ValueError("Invalid time order")
    if len(t) != len(d["positions"]) or len(t) != len(d["branches"]) + 1:
        raise ValueError("Invalid mesh lengths")
    if not np.all(np.isfinite(d["positions"])):
        raise ValueError("Nonfinite positions")
    if not np.all(np.isfinite(t)) or not np.isfinite(d.get("cost", np.nan)):
        raise ValueError("Nonfinite times or fuel")
