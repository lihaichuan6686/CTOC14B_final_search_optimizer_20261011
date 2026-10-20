"""优化所用动力学：Hermite 星历、Lambert、J₂ 加速度、状态转移矩阵和端点打靶。

单位 km、s、km/s。J₂ 的 z 轴必须是标定后的地球极轴，主程序负责旋转星历。
propagate 同时积分 6 维状态和 6×6 STM；shoot 用 STM 修正初速度使末端位置闭合。
"""

import math
import numpy as np
from numba import njit

MU = 398600.4415
RE = 6378.1363
J2 = 0.0010826266835531513


@njit(cache=True)
def norm(v):
    return math.sqrt(np.dot(v, v))


@njit(cache=True)
def target_state(states, target, t):
    n = min(int(t / 60.0), states.shape[1] - 2)
    u = (t - n * 60.0) / 60.0
    r0 = states[target, n, :3]
    v0 = states[target, n, 3:]
    r1 = states[target, n + 1, :3]
    v1 = states[target, n + 1, 3:]
    r = (
        (2 * u**3 - 3 * u * u + 1) * r0
        + (u**3 - 2 * u * u + u) * 60 * v0
        + (-2 * u**3 + 3 * u * u) * r1
        + (u**3 - u * u) * 60 * v1
    )
    v = (
        (6 * u * u - 6 * u) * r0
        + (3 * u * u - 4 * u + 1) * 60 * v0
        + (-6 * u * u + 6 * u) * r1
        + (3 * u * u - 2 * u) * 60 * v1
    ) / 60
    return (r, v)


@njit(cache=True)
def tof_x(x, ll, revolutions):
    q = 1.0 - x * x
    if abs(q) < 1e-08:
        x += 2e-08
        q = 1 - x * x
    y = math.sqrt(max(0.0, 1 - ll * ll * q))
    if x < 1:
        psi = math.acos(max(-1.0, min(1.0, x * y + ll * q)))
    else:
        psi = math.acosh(max(1.0, x * y - ll * -q))
    return ((psi + revolutions * math.pi) / math.sqrt(abs(q)) - x + ll * y) / q


@njit(cache=True)
def lambert(r1, r2, duration, major=0, revolutions=0, side=0):
    nan = np.full(3, np.nan)
    if duration <= 0:
        return (nan, nan)
    rn1, rn2 = (norm(r1), norm(r2))
    chord = norm(r2 - r1)
    s = (rn1 + rn2 + chord) / 2
    normal = np.cross(r1, r2)
    nn = norm(normal)
    if nn < 1e-08 or chord < 1e-08:
        return (nan, nan)
    normal /= nn
    ll = math.sqrt(max(0.0, 1 - chord / s))
    if major:
        ll = -ll
        normal = -normal
    T = math.sqrt(2 * MU / s**3) * duration
    lo, hi = (-1 + 1e-08, 1 - 1e-08)
    increasing = False
    if revolutions == 0:
        hi = 2.0
        while tof_x(hi, ll, 0) > T and hi < 1000000.0:
            hi *= 2
    else:
        a, b = (-0.999999, 0.999999)
        for k in range(45):
            p = a + (b - a) * 0.3819660112501051
            q = a + (b - a) * 0.6180339887498949
            if tof_x(p, ll, revolutions) < tof_x(q, ll, revolutions):
                b = q
            else:
                a = p
        minimum = (a + b) / 2
        if T < tof_x(minimum, ll, revolutions):
            return (nan, nan)
        if side:
            lo = minimum
            increasing = True
        else:
            hi = minimum
    for k in range(48):
        x = (lo + hi) / 2
        tx = tof_x(x, ll, revolutions)
        if (tx > T) != increasing:
            lo = x
        else:
            hi = x
    x = (lo + hi) / 2
    y = math.sqrt(max(0.0, 1 - ll * ll * (1 - x * x)))
    gamma = math.sqrt(MU * s / 2)
    rho = (rn1 - rn2) / chord
    sigma = math.sqrt(max(0.0, 1 - rho * rho))
    vr1 = gamma * (ll * y - x - rho * (ll * y + x)) / rn1
    vr2 = -gamma * (ll * y - x + rho * (ll * y + x)) / rn2
    vt = gamma * sigma * (y + ll * x)
    ir1, ir2 = (r1 / rn1, r2 / rn2)
    return (
        vr1 * ir1 + vt / rn1 * np.cross(normal, ir1),
        vr2 * ir2 + vt / rn2 * np.cross(normal, ir2),
    )


@njit(cache=True)
def arc_min_radius(r1, v1, r2, v2, duration):
    rn = norm(r1)
    h = np.cross(r1, v1)
    ev = np.cross(v1, h) / MU - r1 / rn
    e = norm(ev)
    peri = np.dot(h, h) / (MU * (1 + e))
    inva = 2 / rn - np.dot(v1, v1) / MU
    minimum = min(rn, norm(r2))
    if e < 1e-09:
        return min(minimum, peri)
    if inva > 0:
        a = 1 / inva
        sinE = np.dot(r1, v1) / (e * math.sqrt(MU * a))
        cosE = (1 - rn / a) / e
        mean = (math.atan2(sinE, cosE) - e * sinE) % (2 * math.pi)
        until = (2 * math.pi - mean) / math.sqrt(MU / a**3)
        if duration + 1e-05 >= until:
            minimum = min(minimum, peri)
    elif np.dot(r1, v1) < 0 and np.dot(r2, v2) > 0:
        minimum = min(minimum, peri)
    return minimum


@njit(cache=True)
def rotate(v, axis, angle):
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )


def elements(state):
    r, v = (np.asarray(state[:3]), np.asarray(state[3:]))
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    return {
        "a_km": float(1 / (2 / rn - np.dot(v, v) / MU)),
        "e": float(np.linalg.norm(np.cross(v, h) / MU - r / rn)),
        "inclination_deg": float(np.degrees(np.arccos(h[2] / np.linalg.norm(h)))),
    }


@njit(cache=True)
def acceleration_hessian(r):
    rr = np.dot(r, r)
    rn = math.sqrt(rr)
    z = r[2]
    q = 5 * z * z / rr
    k = 1.5 * J2 * MU * RE * RE
    a = np.empty(3)
    A = np.empty((3, 3))
    for i in range(3):
        c = 3.0 if i == 2 else 1.0
        a[i] = -MU * r[i] / rn**3 + k * r[i] * (q - c) / rn**5
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            dq = (10 * z / rr if j == 2 else 0.0) - 10 * z * z * r[j] / (rr * rr)
            A[i, j] = -MU * (delta / rn**3 - 3 * r[i] * r[j] / rn**5) + k * (
                delta * (q - c) / rn**5 + r[i] * dq / rn**5 - 5 * r[i] * (q - c) * r[j] / rn**7
            )
    return (a, A)


@njit(cache=True)
def variational_rhs(y):
    f = np.empty(42)
    f[:3] = y[3:6]
    acc, A = acceleration_hessian(y[:3])
    f[3:6] = acc
    for i in range(3):
        for j in range(6):
            f[6 + i * 6 + j] = y[6 + (i + 3) * 6 + j]
            f[6 + (i + 3) * 6 + j] = 0.0
            for k in range(3):
                f[6 + (i + 3) * 6 + j] += A[i, k] * y[6 + k * 6 + j]
    return f


@njit(cache=True)
def propagate(r, v, dt, scale=0.012):
    y = np.zeros(42)
    y[:3] = r
    y[3:6] = v
    for i in range(6):
        y[6 + i * 6 + i] = 1.0
    t = 0.0
    minimum = norm(r)
    while t < dt - 1e-09:
        h = min(dt - t, min(90.0, max(1.0, scale * math.sqrt(norm(y[:3]) ** 3 / MU))))
        k1 = variational_rhs(y)
        k2 = variational_rhs(y + h * k1 / 2)
        k3 = variational_rhs(y + h * k2 / 2)
        k4 = variational_rhs(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
        minimum = min(minimum, norm(y[:3]))
    return (y[:6], y[6:].reshape(6, 6), minimum)


@njit(cache=True)
def shoot(r0, r1, dt, branch):
    vi, _ = lambert(r0, r1, dt, branch[0], branch[1], branch[2])
    if not np.isfinite(vi[0]):
        return (vi, vi, np.eye(6), 0.0, 1000000000.0)
    error = 1000000000.0
    minimum = 0.0
    phi = np.eye(6)
    out = np.zeros(6)
    for _ in range(16):
        out, phi, minimum = propagate(r0, vi, dt)
        residual = r1 - out[:3]
        error = norm(residual)
        if error < 2e-07:
            return (vi, out[3:], phi, minimum, error)
        correction = np.linalg.solve(phi[:3, 3:], residual)
        if norm(correction) > 1.0:
            correction /= norm(correction)
        vi += correction
    out, phi, minimum = propagate(r0, vi, dt)
    error = norm(r1 - out[:3])
    return (vi, out[3:], phi, minimum, error)
