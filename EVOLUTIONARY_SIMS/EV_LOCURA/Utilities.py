from __future__ import annotations
import math
import numpy as np

from Params import EPS_DIR


def sigmoid(z: float) -> float:
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def random_unit_vector(rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng()
    ang = rng.uniform(0.0, 2.0 * math.pi)
    return math.cos(ang), math.sin(ang)


def unit_vec(x: float, y: float, eps: float = 1e-9) -> tuple[float, float]:
    n = math.hypot(x, y)
    if n <= eps:
        return 0.0, 0.0
    return x / n, y / n


def torus_delta(target: float, source: float, size: float) -> float:
    """
    Smallest signed displacement from source -> target on a torus of given size.
    """
    d = (target - source) % size
    if d > size / 2:
        d -= size
    return d


def torus_dist2(xs: np.ndarray, ys: np.ndarray, x: float, y: float, size_x: float, size_y: float) -> np.ndarray:
    """
    Squared torus distances from point (x,y) to arrays (xs, ys).
    """
    dx = (xs - x) % size_x
    dx = np.where(dx > size_x / 2, dx - size_x, dx)

    dy = (ys - y) % size_y
    dy = np.where(dy > size_y / 2, dy - size_y, dy)

    return dx * dx + dy * dy


def torus_delta_vec(xs: np.ndarray, ys: np.ndarray, x: float, y: float, size_x: float, size_y: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized torus deltas from (x,y) -> (xs,ys). Returns dx, dy arrays.
    """
    dx = (xs - x) % size_x
    dx = np.where(dx > size_x / 2, dx - size_x, dx)

    dy = (ys - y) % size_y
    dy = np.where(dy > size_y / 2, dy - size_y, dy)

    return dx, dy


def exp_kernel(d: np.ndarray, ell: float) -> np.ndarray:
    if ell <= 1e-9:
        return np.zeros_like(d)
    return np.exp(-d / ell)


def gaussian_repulsion(d: np.ndarray, r: float) -> np.ndarray:
    if r <= 1e-9:
        return np.zeros_like(d)
    return np.exp(- (d / r) ** 2)


def wrap_pos(x: float, y: float, size_x: float, size_y: float) -> tuple[float, float]:
    return x % size_x, y % size_y


def safe_norm(x: float, y: float, eps: float = 1e-9) -> float:
    return float(math.hypot(x, y) + eps)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b <= 0:
        return default
    return a / b


def fmt_pct(x: float, nd: int = 1) -> str:
    return f"{100.0 * x:.{nd}f}%"
