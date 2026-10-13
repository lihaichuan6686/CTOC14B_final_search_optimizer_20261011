'Legacy trajectory container and physical chain evaluation; no beam constructor.'

from dataclasses import dataclass

import json

import math

import os

import time

from pathlib import Path

import numpy as np

from numba import njit

from scipy.optimize import minimize

from .physics import MU, RE, norm, target_state, lambert, arc_min_radius, rotate

@dataclass
class Route:
    ids: list
    times: list
    branches: list
    initial: list
    cost: float
    velocity: list
    nodes: list = None

    def save(self, path):
        data = dict(self.__dict__)
        data.update(
            {
                "units": "km, s, km/s",
                "status": "unverified_search_candidate",
                "count": len(set(self.ids)),
            }
        )
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)

def load_route(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Route(**{k: data[k] for k in Route.__dataclass_fields__ if k in data})

@njit(cache=True)
def chain_cost(states, ids, times, branches, initial, return_details=False):
    r = initial[:3].copy()
    v = initial[3:].copy()
    previous = 0.0
    cost = 0.0
    minr = 1e30
    dvs = np.zeros((len(ids), 3))
    for i in range(len(ids)):
        duration = times[i] - previous
        if duration < 60 or times[i] > 864000:
            return (100000.0 + abs(duration), dvs, np.zeros(3), 0.0)
        r2, _ = target_state(states, ids[i], times[i])
        vi, vf = lambert(r, r2, duration, branches[i, 0], branches[i, 1], branches[i, 2])
        if not math.isfinite(vi[0]):
            return (1000000.0, dvs, np.zeros(3), 0.0)
        rad = arc_min_radius(r, vi, r2, vf, duration)
        minr = min(minr, rad)
        if rad < RE + 205:
            cost += 100 + (RE + 205 - rad) * 0.05
        dvs[i] = vi - v
        cost += norm(dvs[i])
        r, v = (r2, vf)
        previous = times[i]
    return (cost, dvs, v, minr)
