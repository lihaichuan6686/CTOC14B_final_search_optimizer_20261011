"""第一阶段动力学内核，单位为 km、s、km/s。

目标端点由 J2 星历三次 Hermite 插值得到；转移弧使用二体 Lambert，仅用于粗搜排序。
正式结果必须在后续阶段进行 J2 同伦优化和 ATK 验证。
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

MU = 398600.4415
RE = 6378.1363


@njit(cache=True)
def norm(vector: np.ndarray) -> float:
    return math.sqrt(np.dot(vector, vector))


@njit(cache=True)
def target_state(states: np.ndarray, target: int, epoch_s: float):
    """在 60 s J2 星历网格上做三次 Hermite 插值。target 为 0-based。"""
    epoch_s = min(max(epoch_s, 0.0), (states.shape[1] - 1) * 60.0)
    index = min(int(epoch_s / 60.0), states.shape[1] - 2)
    fraction = (epoch_s - index * 60.0) / 60.0
    r0 = states[target, index, :3]
    v0 = states[target, index, 3:]
    r1 = states[target, index + 1, :3]
    v1 = states[target, index + 1, 3:]
    r = (
        (2 * fraction**3 - 3 * fraction**2 + 1) * r0
        + (fraction**3 - 2 * fraction**2 + fraction) * 60.0 * v0
        + (-2 * fraction**3 + 3 * fraction**2) * r1
        + (fraction**3 - fraction**2) * 60.0 * v1
    )
    v = (
        (6 * fraction**2 - 6 * fraction) * r0
        + (3 * fraction**2 - 4 * fraction + 1) * 60.0 * v0
        + (-6 * fraction**2 + 6 * fraction) * r1
        + (3 * fraction**2 - 2 * fraction) * 60.0 * v1
    ) / 60.0
    return r, v


@njit(cache=True)
def _tof_x(x: float, ll: float, revolutions: int) -> float:
    q = 1.0 - x * x
    if abs(q) < 1e-8:
        x += 2e-8
        q = 1.0 - x * x
    y = math.sqrt(max(0.0, 1.0 - ll * ll * q))
    if x < 1.0:
        psi = math.acos(max(-1.0, min(1.0, x * y + ll * q)))
    else:
        psi = math.acosh(max(1.0, x * y - ll * (-q)))
    return ((psi + revolutions * math.pi) / math.sqrt(abs(q)) - x + ll * y) / q


@njit(cache=True)
def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    duration_s: float,
    major: int = 0,
    revolutions: int = 0,
    side: int = 0,
):
    """Izzo 变量形式的 Lambert 求解器。返回出发和到达速度。"""
    nan = np.full(3, np.nan)
    if duration_s <= 0.0:
        return nan, nan
    radius1, radius2 = norm(r1), norm(r2)
    chord = norm(r2 - r1)
    semiperimeter = (radius1 + radius2 + chord) / 2.0
    normal = np.cross(r1, r2)
    normal_norm = norm(normal)
    if normal_norm < 1e-8 or chord < 1e-8:
        return nan, nan
    normal /= normal_norm
    ll = math.sqrt(max(0.0, 1.0 - chord / semiperimeter))
    if major:
        ll = -ll
        normal = -normal
    nondimensional_tof = math.sqrt(2.0 * MU / semiperimeter**3) * duration_s
    lower, upper = -1.0 + 1e-8, 1.0 - 1e-8
    increasing = False
    if revolutions == 0:
        upper = 2.0
        while _tof_x(upper, ll, 0) > nondimensional_tof and upper < 1e6:
            upper *= 2.0
    else:
        a, b = -0.999999, 0.999999
        for _ in range(45):
            left = a + (b - a) * 0.3819660112501051
            right = a + (b - a) * 0.6180339887498949
            if _tof_x(left, ll, revolutions) < _tof_x(right, ll, revolutions):
                b = right
            else:
                a = left
        minimum = (a + b) / 2.0
        if nondimensional_tof < _tof_x(minimum, ll, revolutions):
            return nan, nan
        if side:
            lower = minimum
            increasing = True
        else:
            upper = minimum
    for _ in range(48):
        x = (lower + upper) / 2.0
        value = _tof_x(x, ll, revolutions)
        if (value > nondimensional_tof) != increasing:
            lower = x
        else:
            upper = x
    x = (lower + upper) / 2.0
    y = math.sqrt(max(0.0, 1.0 - ll * ll * (1.0 - x * x)))
    gamma = math.sqrt(MU * semiperimeter / 2.0)
    rho = (radius1 - radius2) / chord
    sigma = math.sqrt(max(0.0, 1.0 - rho * rho))
    vr1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / radius1
    vr2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / radius2
    vt = gamma * sigma * (y + ll * x)
    ir1, ir2 = r1 / radius1, r2 / radius2
    return (
        vr1 * ir1 + vt / radius1 * np.cross(normal, ir1),
        vr2 * ir2 + vt / radius2 * np.cross(normal, ir2),
    )


@njit(cache=True)
def arc_min_radius(r1, v1, r2, v2, duration_s: float) -> float:
    """返回二体弧在给定飞行区间中的最小地心半径。"""
    radius = norm(r1)
    angular_momentum = np.cross(r1, v1)
    eccentricity_vector = np.cross(v1, angular_momentum) / MU - r1 / radius
    eccentricity = norm(eccentricity_vector)
    periapsis = np.dot(angular_momentum, angular_momentum) / (MU * (1.0 + eccentricity))
    inverse_a = 2.0 / radius - np.dot(v1, v1) / MU
    minimum = min(radius, norm(r2))
    if eccentricity < 1e-9:
        return min(minimum, periapsis)
    if inverse_a > 0.0:
        semimajor_axis = 1.0 / inverse_a
        sin_eccentric = np.dot(r1, v1) / (eccentricity * math.sqrt(MU * semimajor_axis))
        cos_eccentric = (1.0 - radius / semimajor_axis) / eccentricity
        mean_anomaly = (math.atan2(sin_eccentric, cos_eccentric) - eccentricity * sin_eccentric) % (
            2.0 * math.pi
        )
        until_periapsis = (2.0 * math.pi - mean_anomaly) / math.sqrt(MU / semimajor_axis**3)
        if duration_s + 1e-5 >= until_periapsis:
            minimum = min(minimum, periapsis)
    elif np.dot(r1, v1) < 0.0 and np.dot(r2, v2) > 0.0:
        minimum = min(minimum, periapsis)
    return minimum


@njit(cache=True)
def rotate(vector, axis, angle):
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
    )
