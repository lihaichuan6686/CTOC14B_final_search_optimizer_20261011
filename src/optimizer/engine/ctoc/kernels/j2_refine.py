"""J2 multiple shooting with state-transition derivatives of each transfer."""

import math
import numpy as np
from numba import njit
from .physics import MU, RE, norm, lambert

J2 = 0.0010826266835531513


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
