"""Explicit SI exchange boundaries and native ATK construction recipes."""

import csv
import datetime as dt
from pathlib import Path
import sys
import numpy as np
from .storage import read_json, atomic_json, digest
from .trajectory import normalize, from_route, rotate
from .evaluation import evaluate
from .kernels.physics import target_state, lambert, MU

EPOCH = dt.datetime(2035, 1, 1, 12, tzinfo=dt.timezone.utc)


def utc(seconds):
    return (
        (EPOCH + dt.timedelta(seconds=float(seconds)))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc(value):
    text = value.strip().removesuffix(" UTC")
    try:
        stamp = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            stamp = dt.datetime.strptime(text, "%d %b %Y %H:%M:%S.%f")
        except ValueError:
            stamp = dt.datetime.strptime(text, "%d %b %Y %H:%M:%S")
    return (
        stamp.replace(tzinfo=dt.timezone.utc)
        if stamp.tzinfo is None
        else stamp.astimezone(dt.timezone.utc)
    )


def initial_state(state):
    if state.get("frame") != "ICRF" or parse_utc(state["epoch_utc"]) != EPOCH:
        raise ValueError("Initial epoch/frame must be the CTOC14B epoch in ICRF")
    if state["type"] == "cartesian":
        return np.r_[state["position_m"], state["velocity_m_s"]] / 1000.0
    if state["type"] != "classical":
        raise ValueError("Unsupported initial state")
    a = state["semi_major_axis_m"] / 1000.0
    e = state["eccentricity"]
    inc, raan, arg, f = np.radians(
        [
            state[k]
            for k in (
                "inclination_deg",
                "raan_deg",
                "argument_of_periapsis_deg",
                "true_anomaly_deg",
            )
        ]
    )
    if a <= 0 or not 0 <= e < 1:
        raise ValueError("Invalid classical orbit")
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(f)) * np.array([np.cos(f), np.sin(f), 0.0])
    v = np.sqrt(MU / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])

    def rz(a):
        return np.array(
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
        )

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(inc), -np.sin(inc)], [0.0, np.sin(inc), np.cos(inc)]]
    )
    R = rz(raan) @ rx @ rz(arg)
    return np.r_[R @ r, R @ v]


def target_id(value):
    if isinstance(value, bool):
        raise ValueError("Invalid target")
    text = str(value).removeprefix("Target")
    if not text.isdigit() or not 1 <= int(text) <= 35:
        raise ValueError("Target must be 1..35")
    return int(text) - 1


def import_solution(path, problem):
    d = read_json(path)
    if "trajectory" in d:
        d = d["trajectory"]
    if "node_ids" in d:
        return normalize(d)
    if (
        d.get("schema_version") != "1.0"
        or d.get("problem_id") != "CTOC14B"
        or not isinstance(d.get("solution_id"), str)
        or not d["solution_id"].strip()
    ):
        raise ValueError("Expected internal trajectory or CTOC14B schema 1.0")
    init = initial_state(d["initial_state"])
    positions = [init[:3].tolist()]
    times = [0.0]
    ids = [-1]
    branches = []
    if "legs_multi" in d:
        if not len(d["seq"]) == len(d["times"]) == len(d["legs_multi"]):
            raise ValueError("Mismatched multi-leg arrays")
        model = "j2" if d.get("j2_report", {}).get("j2_arcs") else "two_body"
        for number, (target, arrival, leg) in enumerate(zip(d["seq"], d["times"], d["legs_multi"])):
            nt = leg["nodes_t"]
            nr = np.asarray(leg["nodes_r"]) / 1000.0
            arcs = leg["arcs"]
            if not len(nt) == len(nr) == len(arcs) + 1 or len(leg["ways"]) != len(arcs):
                raise ValueError("Invalid multi-leg dimensions")
            if (
                abs(nt[0] - times[-1]) > 1e-6
                or abs(nt[-1] - arrival) > 1e-6
                or not np.allclose(nr[0], positions[-1], atol=1e-8, rtol=0)
            ):
                raise ValueError("Discontinuous multi-leg nodes")
            velocities = d["j2_report"]["j2_arcs"][number] if model == "j2" else arcs
            for j in range(len(arcs)):
                expected = np.asarray(velocities[j][0]) / 1000.0
                choices = []
                for rev in range(9):
                    for major in range(2):
                        for side in range(1 if rev == 0 else 2):
                            vi, _ = lambert(nr[j], nr[j + 1], nt[j + 1] - nt[j], major, rev, side)
                            if np.isfinite(vi).all():
                                choices.append((np.linalg.norm(vi - expected), [major, rev, side]))
                if not choices:
                    raise ValueError("No matching multi-leg Lambert branch")
                error, branch = min(choices, key=lambda x: x[0])
                if error > 0.2:
                    raise ValueError("Velocity cannot identify a compatible branch")
                positions.append(nr[j + 1].tolist())
                times.append(float(nt[j + 1]))
                ids.append(target_id(target) if j == len(arcs) - 1 else -1)
                branches.append(branch)
        return normalize(
            dict(
                initial=init.tolist(),
                positions=positions,
                times=times,
                node_ids=ids,
                branches=branches,
                model=model,
                frame="ICRF",
            )
        )
    from .kernels.search import Route

    previous = 0.0
    visits = []
    arrivals = []
    for leg in d["legs"]:
        if ("duration_s" in leg) == ("arrival_utc" in leg):
            raise ValueError("Each leg needs exactly one arrival specification")
        arrival = (
            previous + float(leg["duration_s"])
            if "duration_s" in leg
            else (parse_utc(leg["arrival_utc"]) - EPOCH).total_seconds()
        )
        l = leg.get("lambert", {})
        branches.append(
            [
                int(l.get("major_arc", False)),
                int(l.get("revolutions", 0)),
                int(l.get("arc", "short") == "long"),
            ]
        )
        visits.append(target_id(leg["target"]))
        arrivals.append(arrival)
        previous = arrival
    return from_route(
        Route(visits, arrivals, branches, init.tolist(), 0.0, [0.0, 0.0, 0.0]), problem.states
    )


def recipe(data, problem, condition=True):
    d = normalize(data)
    if condition:
        if d["model"] == "j2":
            from .algorithms import to_pole_j2

            d = rotate(to_pole_j2(d, problem), problem.pole.T)
            d["frame"] = "ICRF"
        else:
            from .kernels.free_impulse import conditioned_nodes

            d = conditioned_nodes(d, max_angle=105.0)
    nodes = []
    ids = []
    times = []
    branches = []
    for i, k in enumerate(d["node_ids"][1:], 1):
        node = dict(
            target=k if k >= 0 else None,
            time=d["times"][i],
            branch=d["branches"][i - 1],
            frame="ICRF",
        )
        if k < 0:
            node["position"] = d["positions"][i]
        else:
            node["offset"] = (
                np.array(d["positions"][i]) - target_state(problem.states, k, d["times"][i])[0]
            ).tolist()
            ids.append(k)
            times.append(d["times"][i])
            branches.append(d["branches"][i - 1])
        nodes.append(node)
    return dict(
        ids=ids,
        times=times,
        branches=branches,
        initial=d["initial"],
        cost=float(d.get("cost", 0.0)),
        velocity=[0.0, 0.0, 0.0],
        nodes=nodes,
        units="km,s,km/s",
    )


def schema_path():
    local = Path(__file__).resolve().parents[2] / "data/refined-multi.schema.json"
    return local if local.exists() else Path(sys.prefix) / "share/ctoc/refined-multi.schema.json"


def multi_json(data, problem, report=None):
    d = normalize(data)
    report = report or evaluate(d, problem, strict=True)
    visits = [i for i, k in enumerate(d["node_ids"]) if k >= 0]
    if not visits or visits[-1] != len(d["times"]) - 1:
        raise ValueError(
            "Multi JSON cannot represent impulses after the final visit; use internal bundle"
        )
    init = d["initial"]
    legs = []
    left = 0
    j2_arcs = []
    for right in visits:
        arcs = [
            [
                (np.array(report["vi_km_s"][j]) * 1000).tolist(),
                (np.array(report["vf_km_s"][j]) * 1000).tolist(),
            ]
            for j in range(left, right)
        ]
        legs.append(
            dict(
                nodes_t=d["times"][left : right + 1],
                nodes_r=(np.array(d["positions"][left : right + 1]) * 1000).tolist(),
                ways=[1] * (right - left),
                arcs=arcs,
            )
        )
        j2_arcs.append(arcs)
        left = right
    out = dict(
        schema_version="1.0",
        problem_id="CTOC14B",
        solution_id="ctoc-"
        + __import__("hashlib").sha256(str(d["times"]).encode()).hexdigest()[:12],
        seq=[d["node_ids"][i] + 1 for i in visits],
        times=[d["times"][i] for i in visits],
        initial_state=dict(
            type="cartesian",
            epoch_utc="2035-01-01T12:00:00.000Z",
            frame="ICRF",
            position_m=(np.array(init[:3]) * 1000).tolist(),
            velocity_m_s=(np.array(init[3:]) * 1000).tolist(),
        ),
        first=dict(
            t_dep=0.0,
            r0=(np.array(init[:3]) * 1000).tolist(),
            vd=(np.array(report["vi_km_s"][0]) * 1000).tolist(),
            dv0=float(np.linalg.norm(report["impulses_km_s"][0])),
        ),
        legs_multi=legs,
        multi_meta=dict(
            epoch="2035-01-01 12:00:00 UTC",
            source_units="m, s, m/s",
            mission_end_s=864000.0,
            final_coast_duration_s=864000.0 - d["times"][-1],
            verification_status=report["status"],
            model=d["model"],
            note="Native recipe/internal bundle preserves all endpoint offsets and final coast. Reimported scenes require new validation.",
        ),
        expected=dict(
            total_dv_km_s=report["total_dv_km_s"],
            notes="Numerical expectation, not an official score",
        ),
    )
    if d["model"] == "j2":
        out["j2_report"] = dict(
            j2_dv_km_s=report["total_dv_km_s"],
            min_alt_km=report["min_alt_km"],
            max_miss_mm=report["max_miss_m"] * 1000,
            j2_arcs=j2_arcs,
        )
    import jsonschema

    jsonschema.Draft202012Validator(read_json(schema_path())).validate(out)
    return out


def simple_json(data, problem, report):
    d = normalize(data)
    if any(k < 0 for k in d["node_ids"][1:]):
        raise ValueError("Simple format cannot represent free impulse nodes")
    for i, k in enumerate(d["node_ids"][1:], 1):
        if (
            np.linalg.norm(
                np.array(d["positions"][i]) - target_state(problem.states, k, d["times"][i])[0]
            )
            > 1e-7
        ):
            raise ValueError("Simple format cannot preserve nonzero encounter offsets")
    full = multi_json(d, problem, report)
    return dict(
        schema_version="1.0",
        problem_id="CTOC14B",
        solution_id=full["solution_id"],
        initial_state=full["initial_state"],
        legs=[
            dict(
                target=f"Target{k+1}",
                arrival_utc=utc(t),
                lambert=dict(
                    method="AdaptSpecified",
                    revolutions=b[1],
                    major_arc=bool(b[0]),
                    arc="long" if b[2] else "short",
                    tolerance_m=0.1,
                ),
            )
            for k, t, b in zip(d["node_ids"][1:], d["times"][1:], d["branches"])
        ],
        expected=full["expected"],
    )


def select_candidate(path, model="j2"):
    path = Path(path)
    if path.is_file():
        return path
    from .storage import restore

    state = restore(path / "checkpoint.json")
    items = state["archives"][model]
    if not items:
        raise ValueError(
            f"No complete {model} candidate; incomplete results cannot be exported as a full solution"
        )
    return path / "candidates" / f"{items[0]['id']}.json"


def export_bundle(source, destination, problem, format="multi-json", model="j2"):
    source = select_candidate(source, model)
    d = import_solution(source, problem)
    report = evaluate(d, problem, strict=True)
    if report["errors"] or not report["complete"]:
        raise ValueError("Export requires a complete, numerically valid trajectory")
    out = Path(destination).resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "trajectory.json", d)
    atomic_json(out / "numerical_validation.json", report)
    atomic_json(out / "native_recipe.json", recipe(d, problem))
    if format == "multi-json":
        atomic_json(out / "solution.multi.json", multi_json(d, problem, report))
    elif format in ("json", "csv"):
        simple = simple_json(d, problem, report)
        if format == "json":
            atomic_json(out / "solution.json", simple)
        else:
            atomic_json(out / "initial_state.json", simple["initial_state"])
            with (out / "solution.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["target", "arrival_hour"])
                writer.writerows(
                    (f"Target{k+1}", t / 3600) for k, t in zip(d["node_ids"][1:], d["times"][1:])
                )
    elif format != "internal":
        raise ValueError("Unknown export format")
    atomic_json(
        out / "bundle.json",
        dict(
            version=1,
            source_sha256=digest(source),
            problem_sha256=problem.fingerprint,
            status="numerical_only_pending_ATK",
            files={
                p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "bundle.json"
            },
        ),
    )
    return dict(
        directory=str(out),
        covered=report["covered"],
        dv_km_s=report["total_dv_km_s"],
        official_status="pending",
    )
