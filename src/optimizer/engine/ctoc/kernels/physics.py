"""Independent Lambert and ephemeris screening kernel (km, s, km/s).

Lambert equations: D. Izzo, Revisiting Lambert's problem, arXiv:1403.2705.
This two-body kernel only ranks candidates; ATK provides final J2 trajectories.
"""

import math
import numpy as np
from numba import njit

MU = 398600.4415
RE = 6378.1363


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
