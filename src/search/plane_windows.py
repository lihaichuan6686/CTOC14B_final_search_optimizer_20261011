"""Plane-band windows of the existing 60 s cubic-Hermite target ephemeris.

No Lambert calls here. For each Hermite segment the plane-band condition is
the degree-six polynomial (n.r)^2 - sin(epsilon)^2 (r.r) <= 0. Bernstein
bounds certify whole intervals; ambiguous pieces are bisected. Thus even a
short grazing window between ephemeris samples is retained. Boundaries are
conservative to edge_tolerance seconds; every transfer checks the band again.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _coefficients(r0, v0, r1, v1, normal, sine2):
    b = np.empty((4, 3))
    b[0] = r0
    b[1] = r0 + 20.0 * v0
    b[2] = r1 - 20.0 * v1
    b[3] = r1
    c3 = np.array([1., 3., 3., 1.])
    c6 = np.array([1., 6., 15., 20., 15., 6., 1.])
    g = b @ normal
    result = np.zeros(7)
    for i in range(4):
        for j in range(4):
            result[i+j] += c3[i]*c3[j]/c6[i+j] * (
                g[i]*g[j] - sine2*np.dot(b[i], b[j]))
    return result


@njit(cache=True)
def _split(coeff):
    work = coeff.copy()
    left = np.empty(7)
    right = np.empty(7)
    left[0], right[6] = work[0], work[6]
    for depth in range(1, 7):
        for j in range(7-depth):
            work[j] = (work[j]+work[j+1])*0.5
        left[depth] = work[0]
        right[6-depth] = work[6-depth]
    return left, right


@njit(cache=True)
def _target_windows(states, normal, sine2, edge_tolerance):
    # A degree-six polynomial has at most four negative intervals on a segment.
    out = np.empty((4*len(states)+4, 2))
    count = 0
    stack_c = np.empty((40, 7))
    stack_a = np.empty(40)
    stack_b = np.empty(40)
    for index in range(len(states)-1):
        coeff = _coefficients(states[index,:3], states[index,3:],
                              states[index+1,:3], states[index+1,3:], normal, sine2)
        stack_c[0] = coeff
        stack_a[0], stack_b[0] = index*60., (index+1)*60.
        size = 1
        while size:
            size -= 1
            c = stack_c[size].copy()
            a, b = stack_a[size], stack_b[size]
            if np.min(c) > 0.0:
                continue
            if np.max(c) <= 0.0 or b-a <= edge_tolerance:
                if count and a <= out[count-1,1]+1e-8:
                    out[count-1,1] = b
                else:
                    if count >= len(out):
                        raise ValueError('Plane-window capacity exceeded')
                    out[count,0], out[count,1] = a,b
                    count += 1
                continue
            left,right = _split(c)
            midpoint = (a+b)*0.5
            if size+2 > len(stack_a):
                raise ValueError('Plane-window subdivision stack exceeded')
            # LIFO: process the left interval first so merging stays ordered.
            stack_c[size] = right
            stack_a[size],stack_b[size] = midpoint,b
            stack_c[size+1] = left
            stack_a[size+1],stack_b[size+1] = a,midpoint
            size += 2
    return out[:count].copy()


def build_windows(states, normal, tolerance_deg, edge_tolerance=0.05):
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1)>1e-10:
        raise ValueError('Plane normal must be a finite unit vector')
    if not 0 < tolerance_deg < 90 or not 1e-5 <= edge_tolerance <= 1:
        raise ValueError('Invalid plane tolerance or window edge accuracy')
    sine2 = math.sin(math.radians(tolerance_deg))**2
    each = [_target_windows(np.ascontiguousarray(x), normal, sine2, edge_tolerance)
            for x in states]
    counts = np.array([len(x) for x in each], dtype=np.int64)
    windows = np.zeros((len(states), max(1, int(counts.max())), 2))
    for k,x in enumerate(each):
        windows[k,:len(x)] = x
    return windows,counts


def sample_plane(rng, z_min=-1.0, z_max=1.0):
    # Uniform z=cos(inclination) and azimuth give equal area on the normal sphere.
    # The normal is oriented: n and -n are different initial motion directions.
    if not (-1.0 <= z_min < z_max <= 1.0):
        raise ValueError('Invalid oriented-plane sector bounds')
    z = float(rng.uniform(z_min, z_max))
    azimuth = float(rng.uniform(0.0, 2.0*math.pi))
    radial = math.sqrt(max(0.0, 1.0-z*z))
    normal = np.array([radial*math.cos(azimuth), radial*math.sin(azimuth), z])
    helper = np.array([0.,0.,1.]) if abs(normal[2])<.95 else np.array([1.,0.,0.])
    u = np.cross(helper,normal)
    u /= np.linalg.norm(u)
    return normal,u,np.cross(normal,u)


@njit(cache=True)
def window_cells(windows, count, earliest, latest, max_span):
    """Clip future windows, splitting only wide windows. No window ranking."""
    if latest < earliest:
        return np.empty((0,3))
    capacity = count + int(math.ceil(max(0.,latest-earliest)/max_span)) + 2
    out = np.empty((capacity,3))
    n = 0
    for j in range(count):
        a,b = max(earliest,windows[j,0]),min(latest,windows[j,1])
        if b < a:
            continue
        parts = max(1,int(math.ceil((b-a)/max_span)))
        for part in range(parts):
            lo = a+(b-a)*part/parts
            hi = a+(b-a)*(part+1)/parts
            out[n,0],out[n,1],out[n,2] = lo,hi,j
            n += 1
    return out[:n].copy()
