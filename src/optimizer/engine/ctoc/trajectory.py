"""Lossless internal trajectory format: km, seconds, km/s; zero-based targets."""

import copy
import hashlib
import numpy as np
from .storage import canonical_bytes


def from_route(route, states):
    from .kernels.physics import target_state

    return normalize(
        dict(
            initial=route.initial,
            positions=[route.initial[:3]]
            + [target_state(states, k, t)[0].tolist() for k, t in zip(route.ids, route.times)],
            times=[0.0, *route.times],
            node_ids=[-1, *route.ids],
            branches=route.branches,
            model="two_body",
            frame="ICRF",
        )
    )


def normalize(data):
    d = copy.deepcopy(data)
    if d.get("format_version", 1) != 1:
        raise ValueError("Unsupported internal format version")
    if str(d.get("units", "km,s,km/s")).replace(" ", "") != "km,s,km/s":
        raise ValueError("Internal trajectory requires km,s,km/s; use the SI exchange importer")
    if d.get("mission_end_s", 864000.0) != 864000.0:
        raise ValueError("Incorrect mission end")
    if "epoch_utc" in d:
        from datetime import datetime, timezone

        if datetime.fromisoformat(d["epoch_utc"].replace("Z", "+00:00")) != datetime(
            2035, 1, 1, 12, tzinfo=timezone.utc
        ):
            raise ValueError("Incorrect internal epoch")
    d.setdefault("frame", "ICRF")
    if d["frame"] != "ICRF":
        raise ValueError("Stored trajectory must be in ICRF")
    d.setdefault(
        "model",
        (
            "j2"
            if "j2" in str(d.get("refinement_model", "")).lower() or "fixed_pole_report" in d
            else "two_body"
        ),
    )
    if d["model"] not in ("two_body", "j2"):
        raise ValueError("Unsupported model")
    d["format_version"] = 1
    d["units"] = "km,s,km/s"
    d["epoch_utc"] = "2035-01-01T12:00:00Z"
    d["mission_end_s"] = 864000.0
    r = np.asarray(d["positions"], dtype=float)
    t = np.asarray(d["times"], dtype=float)
    x = np.asarray(d["initial"], dtype=float)
    ids = np.asarray(d["node_ids"])
    b = np.asarray(d["branches"])
    n = len(t) - 1
    if (
        n < 1
        or r.shape != (n + 1, 3)
        or x.shape != (6,)
        or ids.shape != (n + 1,)
        or b.shape != (n, 3)
    ):
        raise ValueError("Inconsistent trajectory dimensions")
    if not all(np.isfinite(a).all() for a in (r, t, x, ids, b)):
        raise ValueError("Non-finite trajectory")
    if not np.array_equal(ids, ids.astype(int)) or not np.array_equal(b, b.astype(int)):
        raise ValueError("Non-integer node or branch")
    if ids[0] != -1 or np.any((ids < -1) | (ids > 34)):
        raise ValueError("Invalid target IDs")
    visits = ids[ids >= 0]
    if len(visits) != len(set(visits.tolist())):
        raise ValueError("Duplicate target")
    if t[0] != 0 or np.any(np.diff(t) <= 0) or t[-1] > 864000:
        raise ValueError("Non-monotonic or out-of-window epochs")
    if np.max(np.abs(r[0] - x[:3])) > 1e-9:
        raise ValueError("Initial position mismatch")
    if (
        np.any(b[:, 0] < 0)
        | np.any(b[:, 0] > 1)
        | np.any(b[:, 1] < 0)
        | np.any(b[:, 2] < 0)
        | np.any(b[:, 2] > 1)
    ):
        raise ValueError("Invalid Lambert branch")
    d.update(
        initial=x.tolist(),
        positions=r.tolist(),
        times=t.tolist(),
        node_ids=ids.astype(int).tolist(),
        branches=b.astype(int).tolist(),
    )
    indices = np.flatnonzero(ids >= 0)
    # Numerical kernels use this only for the initial-coordinate parameterization.
    d["source_route"] = dict(
        ids=ids[indices].astype(int).tolist(),
        times=t[indices].tolist(),
        branches=[d["branches"][i - 1] for i in indices],
        initial=x.tolist(),
        cost=float(d.get("cost", 0.0)),
        velocity=[0.0, 0.0, 0.0],
        nodes=None,
    )
    return d


def identity(data):
    keys = ("initial", "positions", "times", "node_ids", "branches", "model", "frame")
    return hashlib.sha256(canonical_bytes({k: data[k] for k in keys})).hexdigest()


def family(data):
    visits = [k for k in data["node_ids"] if k >= 0]
    return hashlib.sha256(canonical_bytes(visits)).hexdigest()[:16]


def rotate(data, matrix):
    d = copy.deepcopy(data)
    d["positions"] = (np.asarray(d["positions"]) @ matrix).tolist()
    initial = np.array(d["initial"])
    d["initial"] = np.r_[initial[:3] @ matrix, initial[3:] @ matrix].tolist()
    d.pop("encounter_offsets_km", None)
    return d
