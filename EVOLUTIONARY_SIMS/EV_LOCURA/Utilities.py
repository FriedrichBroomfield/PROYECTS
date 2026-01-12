# Utilities.py
from __future__ import annotations

import math
import numpy as np
from Params import *

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def random_unit_vector(rng: np.random.Generator) -> tuple[float, float]:
    ang = rng.uniform(0.0, 2.0 * math.pi)
    return math.cos(ang), math.sin(ang)

def unit_vec(x: float, y: float, eps: float = 1e-9) -> tuple[float, float]:
    n = math.hypot(x, y)
    if n <= eps:
        return 0.0, 0.0
    return x / n, y / n

def safe_norm(x: float, y: float, eps: float = 1e-9) -> float:
    return float(math.hypot(x, y) + eps)

def wrap_pos(x: float, y: float, size_x: float, size_y: float) -> tuple[float, float]:
    return x % size_x, y % size_y

def torus_delta(target: float, source: float, size: float) -> float:
    d = (target - source) % size
    if d > size / 2:
        d -= size
    return d

def torus_delta_vec(xs: np.ndarray, ys: np.ndarray, x: float, y: float, size_x: float, size_y: float) -> tuple[np.ndarray, np.ndarray]:
    dx = (xs - x) % size_x
    dx = np.where(dx > size_x / 2, dx - size_x, dx)
    dy = (ys - y) % size_y
    dy = np.where(dy > size_y / 2, dy - size_y, dy)
    return dx, dy

def torus_dist2(xs: np.ndarray, ys: np.ndarray, x: float, y: float, size_x: float, size_y: float) -> np.ndarray:
    dx, dy = torus_delta_vec(xs, ys, x, y, size_x, size_y)
    return dx * dx + dy * dy

def exp_kernel(d: np.ndarray, ell: float) -> np.ndarray:
    if ell <= 1e-9:
        return np.zeros_like(d)
    return np.exp(-d / ell)

def gaussian_repulsion(d: np.ndarray, r: float) -> np.ndarray:
    if r <= 1e-9:
        return np.zeros_like(d)
    return np.exp(- (d / r) ** 2)

# ----------------------------
# Energy aggregator E(w,f) + marginales
# ----------------------------
def E_of(w: float, f: float) -> float:
    w = max(w, EPS)
    f = max(f, EPS)
    if USE_CES:
        inside = CES_alpha * (w ** CES_rho) + (1.0 - CES_alpha) * (f ** CES_rho)
        return inside ** (1.0 / CES_rho)
    return CD_A * (w ** CD_a) * (f ** CD_b)

def dE_dw(w: float, f: float) -> float:
    w = max(w, EPS); f = max(f, EPS)
    if USE_CES:
        inside = CES_alpha * (w ** CES_rho) + (1.0 - CES_alpha) * (f ** CES_rho)
        return CES_alpha * (w ** (CES_rho - 1.0)) * (inside ** (1.0 / CES_rho - 1.0))
    return (CD_a / w) * E_of(w, f)

def dE_df(w: float, f: float) -> float:
    w = max(w, EPS); f = max(f, EPS)
    if USE_CES:
        inside = CES_alpha * (w ** CES_rho) + (1.0 - CES_alpha) * (f ** CES_rho)
        return (1.0 - CES_alpha) * (f ** (CES_rho - 1.0)) * (inside ** (1.0 / CES_rho - 1.0))
    return (CD_b / f) * E_of(w, f)

# ----------------------------
# Collision: uniform grid (spatial hashing), torus-aware
# ----------------------------
def _torus_dx(dx: float, size: float) -> float:
    if dx > size / 2:
        dx -= size
    elif dx < -size / 2:
        dx += size
    return dx

def resolve_collisions_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    size_x: float,
    size_y: float,
    r_col: float,
    cell_size: float,
    n_iters: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard-sphere: asegura distancia mínima r_col entre centros.
    Complejidad ~ O(N) esperada.
    """
    N = xs.size
    if N <= 1:
        return xs, ys

    L = max(float(cell_size), 1e-6)
    nx = int(math.ceil(size_x / L))
    ny = int(math.ceil(size_y / L))

    r2 = r_col * r_col
    eps = 1e-9

    x = xs.astype(float, copy=True)
    y = ys.astype(float, copy=True)

    for _ in range(max(1, int(n_iters))):
        buckets: dict[tuple[int, int], list[int]] = {}
        ix = (np.floor(x / L).astype(int)) % nx
        iy = (np.floor(y / L).astype(int)) % ny

        for i in range(N):
            key = (int(ix[i]), int(iy[i]))
            buckets.setdefault(key, []).append(i)

        for (cx, cy), idxs in buckets.items():
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    k = ((cx + ox) % nx, (cy + oy) % ny)
                    jdxs = buckets.get(k)
                    if not jdxs:
                        continue

                    for i in idxs:
                        for j in jdxs:
                            if j <= i:
                                continue

                            dx = _torus_dx(x[j] - x[i], size_x)
                            dy = _torus_dx(y[j] - y[i], size_y)
                            d2 = dx * dx + dy * dy
                            if d2 < r2:
                                d = math.sqrt(d2 + eps)
                                ux = dx / d
                                uy = dy / d
                                overlap = r_col - d
                                shift = 0.5 * overlap

                                x[i] = (x[i] - shift * ux) % size_x
                                y[i] = (y[i] - shift * uy) % size_y
                                x[j] = (x[j] + shift * ux) % size_x
                                y[j] = (y[j] + shift * uy) % size_y

    return x, y
