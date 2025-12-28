import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


# ============================================================
# PARAMS (esenciales)
# ============================================================
P = dict(
    # World
    R_MAX=10.0,

    # Population
    POP_CAP=2000,
    N0=500,
    HACK_RATIO0=0.35,         # 1=Hawk(Hack), 0=Dove(Dover)

    # Energy
    E_INIT_MIN=80.0,
    E_INIT_MAX=120.0,

    # Coordinated day/night
    DAY_STEPS=90,
    NIGHT_STEPS=60,           # recomendado >= R_MAX/SPEED
    NEST_THICKNESS=0.25,

    # Food (unit, 1-day life)
    FOOD_PER_DAY=1400,
    FOOD_VALUE_V=40.0,        # energía por manzana (V)
    FOOD_MARGIN=0.6,
    FOOD_SPAWN_R_MIN=1.2,     # sube para evitar centro
    GRAB_RADIUS=0.42,

    # Sensing + motion
    SENSORY_RANGE=2.2,
    SENSORY_FACTOR=5.0,
    SPEED=0.22,
    BROWNIAN_SIGMA=0.35,

    # Costs
    METABOLIC_DAY=0.05*0,       # costo por step (todos)
    METABOLIC_NIGHT=0.03*0,
    MOVE_COST_PER_DIST=0.70*0,  # costo * distancia movida

    # Reproduction (dawn, nest)
    REPRO_THRESHOLD=120.0,    # energía mínima al amanecer para reproducir
    REPRO_SPLIT=0.5,          # padre/hijo se quedan con split * energía del padre

    # Daily goals (prioridad 1 y 2)
    REPRO_GOAL=170.0,         # objetivo durante el día para "salir del mercado" y volver
    SURVIVE_GOAL=60.0,        # si queda poco día y ya estás seguro -> vuelve

    # Energetic satiety (simplificación clave)
    SATIETY_E=130.0,          # techo de energía por ingesta (no recorta, limita absorción)
    SATIETY_EPS=10,         # tolerancia

    # Hawk-Dove
    INJURY_COST_D=21.0,       # D (H-H pierde uno)
    TIME_COST_T=15.0,         # T (D-D ambos)

    # UI
    STEPS_PER_FRAME_INIT=5,
)

TWOPI = 2 * np.pi


def to_cpu(a):
    return cp.asnumpy(a)


# ============================================================
# Geometry & motion helpers
# ============================================================
def clip_to_circle(x, y, r_max: float):
    r = cp.sqrt(x * x + y * y) + 1e-9
    over = r > r_max
    scale = (r_max / r)
    x = cp.where(over, x * scale, x)
    y = cp.where(over, y * scale, y)
    return x, y


def brownian_step(n, sigma, max_step):
    dx = cp.random.standard_normal((n,), dtype=cp.float32) * cp.float32(sigma)
    dy = cp.random.standard_normal((n,), dtype=cp.float32) * cp.float32(sigma)
    d = cp.sqrt(dx * dx + dy * dy) + 1e-9
    scale = cp.minimum(cp.float32(1.0), cp.float32(max_step) / d)
    dx *= scale
    dy *= scale
    step = cp.sqrt(dx * dx + dy * dy)
    return dx, dy, step


def towards_step(x, y, tx, ty, max_step, noise_angle=0.10):
    dx = tx - x
    dy = ty - y
    dist = cp.sqrt(dx * dx + dy * dy) + 1e-9
    step = cp.minimum(cp.full_like(dist, cp.float32(max_step)), dist)
    ux = dx / dist
    uy = dy / dist
    ang = cp.random.standard_normal(dist.shape, dtype=cp.float32) * cp.float32(noise_angle)
    cs = cp.cos(ang)
    sn = cp.sin(ang)
    ux2 = ux * cs - uy * sn
    uy2 = ux * sn + uy * cs
    return step * ux2, step * uy2, step


def radial_out_step(x, y, r_max, max_step):
    r = cp.sqrt(x * x + y * y) + 1e-9
    remaining = cp.float32(r_max) - r
    step = cp.minimum(cp.full_like(r, cp.float32(max_step)), cp.maximum(remaining, 0.0))
    ux = x / r
    uy = y / r
    return step * ux, step * uy, step
