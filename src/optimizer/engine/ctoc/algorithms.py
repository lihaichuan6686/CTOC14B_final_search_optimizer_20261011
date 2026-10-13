'Stage-two numerical operations only. No initial-solution generator.'

import copy

import time

import numpy as np

from .trajectory import from_route, normalize, rotate

from .kernels.physics import RE, target_state

def configure_branches(maxrev):
    # Numba captures globals at compilation; run-specific cache namespaces prevent
    # reusing a specialization from another branch configuration.
    from .kernels import advanced_search, basin_escape, hybrid_repair

    branches = np.array(
        [[m, r, s] for r in range(maxrev + 1) for m in range(2) for s in range(1 if r == 0 else 2)],
        dtype=np.int64,
    )
    for module in (advanced_search, basin_escape, hybrid_repair):
        module.BRANCHES = branches

def to_pole_j2(data, problem):
    from .kernels.free_impulse import conditioned_nodes
    from .kernels.external_search import regularize

    d = normalize(data)
    if d["model"] == "two_body":
        d = conditioned_nodes(d, max_angle=105.0)
    d = rotate(d, problem.pole)
    d["frame"] = "fixed_pole_surrogate"
    d["model"] = "j2"
    return regularize(d)

def refine_j2(data, problem, states_pole, cfg, rng, seconds, topology=False, iteration=0):
    from .kernels.elliptic_reduced_scp import refine
    from .kernels.multi_impulse_round8 import ensure_offsets, insert_coast, remove_burn
    from .kernels.adaptive_scp import coarsen, trajectory

    started = time.monotonic()
    d = to_pole_j2(data, problem)
    ensure_offsets(d, states_pole)
    meta = dict(operator="j2_scp")
    if topology:
        mode = cfg.get('_topology_mode', iteration % (5 if cfg.get("weighted_escape", False) else 4))
        if mode == 0:
            from .kernels.elliptic_global_primer import scan
            from .kernels.impulse_insertion import add_zero_nodes

            peaks, report = scan(d, states_pole, cfg["primer_samples"])
            selected = []
            for p in peaks:
                if p["primer_norm"] < 1.0005:
                    break
                if all(abs(p["arc"] - q["arc"]) > 1 for q in selected):
                    selected.append(p)
                if len(selected) == 4:
                    break
            if selected:
                d = add_zero_nodes(d, selected)
            meta = dict(
                operator="stm_insertion",
                insertions=len(selected),
                max_primer_norm=report["max_primer_norm"],
            )
        elif mode == 1:
            from .kernels.surrogate_pairs import scan, insert_pair

            candidates, _ = scan(d, samples=5, both_ends=True)
            if candidates:
                d = insert_pair(d, candidates[0])
            meta = dict(operator="paired_insertion", opportunities=len(candidates))
        elif mode == 2:
            free = [i for i in range(1, len(d["times"]) - 1) if d["node_ids"][i] < 0]
            if free:
                d = remove_burn(d, int(rng.choice(free)))
            else:
                d = insert_coast(d, int(rng.integers(len(d["branches"]))), 0.5)
            meta = dict(operator="delete_rebuild")
        elif mode == 3:
            d = coarsen(d)
            meta = dict(operator="coarsen")
        else:
            # Engineering continuation heuristic. Weighted values never enter
            # the leaderboard; restore the physical unweighted objective below.
            weights = np.exp(rng.normal(0.0, 0.55, len(d["branches"])))
            weights = np.clip(weights, 0.3, 3.0)
            weights /= weights.mean()
            before = trajectory(d)[4]
            remaining = max(0.01, seconds - (time.monotonic() - started))
            d = refine(d, states_pole, remaining * 0.4 / 60,
                       weights=weights, max_iterations=max(10, cfg["local_iterations"] // 2))
            meta = dict(operator="weighted_objective_escape", original_dv_km_s=before,
                        escape_dv_km_s=trajectory(d)[4], weights=weights.tolist())
    ensure_offsets(d, states_pole)
    d = refine(
        d,
        states_pole,
        max(0.0001, (seconds - (time.monotonic() - started)) / 60),
        max_iterations=cfg["local_iterations"],
    )
    d["cost"] = trajectory(d)[4]
    d = rotate(d, problem.pole.T)
    d["frame"] = "ICRF"
    d["model"] = "j2"
    return normalize(d), meta
