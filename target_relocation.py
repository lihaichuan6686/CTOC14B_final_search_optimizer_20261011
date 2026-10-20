"""解除目标约束后，沿 J₂ 轨迹寻找其他访问时机，再用弹性 SCP 恢复访问。

几何距离仅用于机会筛选，解除约束后的低费用不能当作完整解的成绩。
"""

import copy
import numpy as np
from scipy.optimize import minimize_scalar, linear_sum_assignment
from orbit_dynamics import shoot, propagate, target_state
from impulse_nodes import insert_coast, ensure_offsets, regularize
from j2_optimization import refine, repair


def release_target(trajectory, states, target, seconds, iterations, opportunity_count):
    """只解除一个访问球约束，保留该处自由脉冲，再观察全链的松弛收益。"""
    released = copy.deepcopy(trajectory)
    old_slot = [k for k in released["node_ids"] if k >= 0].index(target)
    node = released["node_ids"].index(target)
    released["node_ids"][node] = -1
    ensure_offsets(released, states)
    released = refine(released, states, seconds / 60, max_iterations=iterations)
    choices = alternatives(released, states, target, old_slot)[:opportunity_count]
    gain = trajectory["cost"] - released["cost"]
    return released, target, gain, choices


def restore_encounters(
    trajectory,
    states,
    targets,
    seconds,
    polish_seconds,
    iterations,
    penalties=(0.0002, 0.002, 0.02, 0.2),
):
    """提高位置违约惩罚，恢复访问球；最后回到无罚项的真实Δv目标。"""
    for penalty in penalties:
        trajectory = repair(trajectory, states, targets, 0.995, penalty, seconds=seconds)
        if trajectory["elastic_excess_km"] < 1e-5:
            break
    if trajectory["elastic_excess_km"] > 0.5:
        raise ValueError("弹性修复未恢复访问约束")
    for node, target in enumerate(trajectory["node_ids"]):
        if target not in targets:
            continue
        center = target_state(states, target, trajectory["times"][node])[0]
        offset = np.array(trajectory["positions"][node]) - center
        distance = np.linalg.norm(offset)
        if distance > 0.995:
            trajectory["positions"][node] = (center + offset * 0.995 / distance).tolist()
    ensure_offsets(trajectory, states)
    return refine(trajectory, states, polish_seconds / 60, max_iterations=iterations)


def relocate_target(
    released, states, target, encounter, repair_seconds, polish_seconds, iterations
):
    times = np.array(released["times"])
    moment = encounter["time_s"]
    arc = int(np.searchsorted(times, moment) - 1)
    fraction = (moment - times[arc]) / (times[arc + 1] - times[arc])
    candidate = insert_coast(released, arc, fraction)
    candidate["node_ids"][arc + 1] = target
    return restore_encounters(
        candidate, states, [target], repair_seconds, polish_seconds, iterations
    )


def rotate_and_reassign(
    trajectory, states, angle_deg, axis_name, repair_seconds=30, polish_seconds=55, iterations=850
):
    """旋转载体轨迹，匈牙利匹配35个目标和弧段，再恢复全部访问约束。"""
    candidate = copy.deepcopy(trajectory)
    initial = np.array(candidate["initial"])
    axis = (
        np.array([0.0, 0.0, 1.0]) if axis_name == "z" else initial[:3] / np.linalg.norm(initial[:3])
    )
    ax, ay, az = axis
    cross_matrix = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    angle = np.deg2rad(angle_deg)
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross_matrix
    )
    candidate["positions"] = (np.array(candidate["positions"]) @ rotation.T).tolist()
    candidate["initial"] = (initial.reshape(2, 3) @ rotation.T).ravel().tolist()
    candidate["node_ids"] = [-1] * len(candidate["node_ids"])
    candidate.pop("encounter_offsets_km", None)
    candidate = regularize(candidate)

    distances = np.full((35, len(candidate["branches"])), 1e12)
    opportunities = {}
    for target in range(35):
        choices = alternatives(candidate, states, target, -999, per_arc=True)
        opportunities[target] = {choice["arc"]: choice for choice in choices}
        for choice in choices:
            distances[target, choice["arc"]] = choice["distance_km"]
    targets, arcs = linear_sum_assignment(distances)
    if len(targets) != 35 or distances[targets, arcs].max() > 1e11:
        raise ValueError("没有覆盖35个目标的几何匹配")
    encounters = [
        (int(target), opportunities[int(target)][int(arc)]) for target, arc in zip(targets, arcs)
    ]
    for target, encounter in sorted(encounters, key=lambda item: item[1]["time_s"]):
        times = np.array(candidate["times"])
        moment = encounter["time_s"]
        arc = int(np.searchsorted(times, moment) - 1)
        candidate = insert_coast(
            candidate, arc, (moment - times[arc]) / (times[arc + 1] - times[arc])
        )
        candidate["node_ids"][arc + 1] = target
    return restore_encounters(
        candidate,
        states,
        list(range(35)),
        repair_seconds,
        polish_seconds,
        iterations,
        (0.0001, 0.001, 0.01, 0.1),
    )


def alternatives(d, states, target, old_slot, per_arc=False):
    """One continuously refined geometric minimum per different visit slot."""
    t = np.array(d["times"])
    r = np.array(d["positions"])
    slots = {}
    for arc in range(len(t) - 1):
        slot = sum((k >= 0 for k in d["node_ids"][1 : arc + 1]))
        if slot == old_slot:
            continue
        duration = t[arc + 1] - t[arc]
        if duration < 50:
            continue
        vi, _, _, _, err = shoot(r[arc], r[arc + 1], duration, np.array(d["branches"][arc]))
        if err > 0.0001:
            continue

        def distance(dt):
            s, _, _ = propagate(r[arc], vi, float(dt))
            return float(
                np.linalg.norm(s[:3] - target_state(states, target, float(t[arc] + dt))[0])
            )

        grid = np.linspace(22, duration - 22, max(4, int(duration / 600) + 1))
        values = np.array([distance(x) for x in grid])
        best = int(np.argmin(values))
        lo = grid[max(0, best - 1)]
        hi = grid[min(len(grid) - 1, best + 1)]
        fit = minimize_scalar(
            distance, bounds=(lo, hi), method="bounded", options=dict(xatol=0.2, maxiter=30)
        )
        dt = float(fit.x)
        miss = float(fit.fun)
        entry = dict(
            arc=arc,
            fraction=dt / duration,
            time_s=float(t[arc] + dt),
            distance_km=miss,
            new_slot=slot,
        )
        key = arc if per_arc else slot
        if key not in slots or miss < slots[key]["distance_km"]:
            slots[key] = entry
    return sorted(slots.values(), key=lambda x: x["distance_km"])
