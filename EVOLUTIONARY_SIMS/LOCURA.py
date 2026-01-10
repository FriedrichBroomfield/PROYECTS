
from __future__ import annotations
# Parameters.py
# -*- coding: utf-8 -*-
"""
Centralized simulation parameters.

This file is intentionally explicit: every parameter used by the simulation
is defined here (no hidden defaults).
"""



# ----------------------------
# World geometry
# ----------------------------
GRID_SIZE_X: int = 100
GRID_SIZE_Y: int = 100

NEST_RADIUS: float = 10.0  # Nest centered at (GRID_SIZE_X/2, GRID_SIZE_Y/2)

# Water sources
N_WATER_SOURCES: int = 4
WATER_SOURCES_RADIUS: float = 6.0  # for visualization + "blob" spawning exclusion
WATER_DRINK_RADIUS: float = 2.0    # must be > SOURCE_REPULSION_RADIUS

# Anti-crowding around water sources (repel near the center)
SOURCE_REPULSION_RADIUS: float = 0.7
THETA_SOURCE_REPULSION: float = 2.0  # strength

# ----------------------------
# Food
# ----------------------------
FOOD_VALUE: float = 12.0
FOOD_EAT_RADIUS: float = 1.5

MAX_FOOD: int = 250
FOOD_SPAWN_PER_STEP: int = 8

# Soft minimum separation between food particles (spawn only)
FOOD_MIN_SEP: float = 1.0

# ----------------------------
# Two types: A (slow, big vision) / B (fast, small vision)
# ----------------------------
TYPE_A: int = 0
TYPE_B: int = 1

# "Rapidez": distance moved per step
STEP_SIZE_A: float = 1.2
STEP_SIZE_B: float = 2.6

# Vision radius (perception radius)
VISION_RADIUS_A: float = 16.0
VISION_RADIUS_B: float = 8.0

# Heading inertia (higher = smoother turns, less jitter)
HEADING_ALPHA_A: float = 0.25
HEADING_ALPHA_B: float = 0.18

# Exploration noise strength when signals cancel (anti-paralysis)
OMEGA0_A: float = 0.25
OMEGA0_B: float = 0.45
G1_CANCEL_SCALE: float = 0.25

# Exploration persistence baseline (fast/low-vision should be more ballistic)
PERSIST_BASE_A: int = 8
PERSIST_BASE_B: int = 16
C_SEEK_PERSIST: float = 2.0  # multiplies (pi_w * L_w)

# ----------------------------
# Kernels / numerics
# ----------------------------
EPS: float = 1e-6
EPS_DIR: float = 1e-9

# Attraction kernel length scale is set per-agent: ell = vision/2
# psi(d) = exp(-d/ell)
# Repulsion kernel (agents, sources): eta(d) = exp(-(d/r)^2)

# ----------------------------
# Needs: urgency weights and satiety gating
# ----------------------------
URGENCY_P: float = 1.0  # U = 1/(resource+eps)^p

# Satiety threshold and slope (AND gate)
ENERGY_SAT: float = 35.0
HYDRATION_SAT: float = 35.0
SAT_SIGMOID_K: float = 0.35

# Water visibility gating
G0_WATER_SIGNAL: float = 0.35  # scale for L_w = g0/(||a_w_vis||+g0)

# Memory (water)
ETA_MEM_UPDATE: float = 0.60
DELTA_MEM_DECAY_A: float = 0.02
DELTA_MEM_DECAY_B: float = 0.03
KAPPA_MEM: float = 1.0
KAPPA_XI: float = 0.7

# ----------------------------
# Social / mate dynamics
# ----------------------------
INTERACTION_RADIUS: float = 1.5  # r_e for mate selection + eating overlap scale

AGENT_REPULSION_RADIUS: float = 1.0
THETA_AGENT_REPULSION_DEFAULT: float = 1.2

# Basal "inconsciente" attraction to same/different type
THETA_SAME_DEFAULT: float = 0.25
THETA_DIFF_DEFAULT: float = 0.15

# Mate attraction strength (only active under joint satiety)
THETA_MATE_DEFAULT: float = 0.60
P_MATE_DEFAULT: float = 0.70  # preference for same-type mates (if same -> p; if diff -> 1-p)

# ----------------------------
# Metabolism / costs
# ----------------------------
# Costs are applied to BOTH resources (energy, hydration).
# This aligns with "each step costs w and f" (here hydration ~ w, energy ~ f).
MOVE_COST_PER_DISTANCE: float = 0.18  # multiplied by step_size; subtracted from both

BASE_ENERGY_DECAY: float = 0.02
BASE_HYDRATION_DECAY: float = 0.01

LOW_HYDRATION_PENALTY_ENERGY: float = 0.02
LOW_HYDRATION_THRESHOLD: float = 3.0

ENERGY_MAX: float = 60.0
HYDRATION_MAX: float = 60.0

# Drinking gain per step if inside drink radius of any source
DRINK_GAIN: float = 3.5

# ----------------------------
# Reproduction (optional, partner-based)
# ----------------------------
ENABLE_REPRODUCTION: bool = True
REPRO_REQUIRE_NEST: bool = False  # if True, both parents must be in nest
REPRO_BETA: float = 0.02          # per-step probability scale
REPRO_COOLDOWN_STEPS: int = 220

REPRO_COST_ENERGY: float = 8.0
REPRO_COST_HYDRATION: float = 8.0

CHILD_SHARE: float = 0.45  # fraction transferred to child from each parent (symmetric)
CHILD_SPAWN_JITTER: float = 1.0  # spawn within [-1,1] box around parent

# ----------------------------
# Visualization
# ----------------------------
PLOT_VMIN_ENERGY: float = 0.0
PLOT_VMAX_ENERGY: float = 60.0




import math
import numpy as np


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
    # psi(d) = exp(-d/ell)
    if ell <= 1e-9:
        return np.zeros_like(d)
    return np.exp(-d / ell)


def gaussian_repulsion(d: np.ndarray, r: float) -> np.ndarray:
    # eta(d) = exp(-(d/r)^2)
    if r <= 1e-9:
        return np.zeros_like(d)
    return np.exp(- (d / r) ** 2)


def wrap_pos(x: float, y: float, size_x: float, size_y: float) -> tuple[float, float]:
    return x % size_x, y % size_y


def safe_norm(x: float, y: float, eps: float = 1e-9) -> float:
    return float(math.hypot(x, y) + eps)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




# ----------------------------
# World
# ----------------------------
class Universe:
    def __init__(
        self,
        size_x: int = GRID_SIZE_X,
        size_y: int = GRID_SIZE_Y,
        food_value: float = FOOD_VALUE,
        nest_radius: float = NEST_RADIUS,
        water_sources_radius: float = WATER_SOURCES_RADIUS,
        rng: np.random.Generator | None = None,
    ):
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.food_value = float(food_value)
        self.nest_radius = float(nest_radius)
        self.water_sources_radius = float(water_sources_radius)
        self.rng = rng or np.random.default_rng()

        self.food_x = np.empty(0, dtype=float)
        self.food_y = np.empty(0, dtype=float)
        self.water_sources = np.empty((0, 2), dtype=float)

    def gen_water_sources(self, n_sources: int = N_WATER_SOURCES):
        """
        Even-ish distribution with separation constraints (torus-aware distances).
        """
        positions: list[tuple[float, float]] = []
        attempts, max_attempts = 0, 5000

        cx, cy = self.size_x / 2, self.size_y / 2
        min_sep2 = (self.water_sources_radius * 2.5) ** 2

        while len(positions) < n_sources and attempts < max_attempts:
            x = self.rng.uniform(0, self.size_x)
            y = self.rng.uniform(0, self.size_y)

            # keep away from nest
            dx = torus_delta(x, cx, self.size_x)
            dy = torus_delta(y, cy, self.size_y)
            if (dx * dx + dy * dy) < (self.nest_radius + self.water_sources_radius) ** 2:
                attempts += 1
                continue

            ok = True
            for wx, wy in positions:
                ddx = torus_delta(x, wx, self.size_x)
                ddy = torus_delta(y, wy, self.size_y)
                if (ddx * ddx + ddy * ddy) < min_sep2:
                    ok = False
                    break

            if ok:
                positions.append((x, y))

            attempts += 1

        # fallback: if we couldn't place enough, force quadrant placement
        if len(positions) < n_sources:
            positions = [
                (self.size_x * 0.25, self.size_y * 0.25),
                (self.size_x * 0.25, self.size_y * 0.75),
                (self.size_x * 0.75, self.size_y * 0.25),
                (self.size_x * 0.75, self.size_y * 0.75),
            ][:n_sources]

        self.water_sources = np.array(positions, dtype=float)

    def spawn_food(self, n_new: int):
        """
        Spawn food uniformly with soft collision checks and avoid water blobs.
        """
        if n_new <= 0:
            return

        fx, fy = [], []
        attempts, max_attempts = 0, max(200, n_new * 40)

        existing_x = self.food_x
        existing_y = self.food_y
        min_sep2 = FOOD_MIN_SEP ** 2

        while len(fx) < n_new and attempts < max_attempts:
            x = self.rng.uniform(0, self.size_x)
            y = self.rng.uniform(0, self.size_y)

            # keep away from water blobs (visual radius)
            if self.water_sources.size:
                d2w = torus_dist2(self.water_sources[:, 0], self.water_sources[:, 1], x, y, self.size_x, self.size_y)
                if np.any(d2w < (self.water_sources_radius ** 2)):
                    attempts += 1
                    continue

            # avoid overlapping food (soft)
            ok = True
            if existing_x.size:
                d2f = torus_dist2(existing_x, existing_y, x, y, self.size_x, self.size_y)
                if np.any(d2f < min_sep2):
                    ok = False

            if ok and fx:
                d2new = torus_dist2(np.array(fx), np.array(fy), x, y, self.size_x, self.size_y)
                if np.any(d2new < min_sep2):
                    ok = False

            if ok:
                fx.append(x)
                fy.append(y)

            attempts += 1

        if fx:
            self.food_x = np.concatenate([self.food_x, np.array(fx, dtype=float)])
            self.food_y = np.concatenate([self.food_y, np.array(fy, dtype=float)])

    def remove_food_index(self, idx: int):
        self.food_x = np.delete(self.food_x, idx)
        self.food_y = np.delete(self.food_y, idx)


# ----------------------------
# Agent
# ----------------------------
class Cell:
    def __init__(
        self,
        x: float,
        y: float,
        cell_type: int,
        energy: float = 18.0,
        hydration: float = 12.0,
        rng: np.random.Generator | None = None,
    ):
        self.x = float(x)
        self.y = float(y)
        self.type = int(cell_type)
        self.rng = rng or np.random.default_rng()

        # Type-dependent movement + perception
        if self.type == TYPE_A:
            self.step = float(STEP_SIZE_A)
            self.vision = float(VISION_RADIUS_A)
            self.alpha_h = float(HEADING_ALPHA_A)
            self.omega0 = float(OMEGA0_A)
            self.persist_base = int(PERSIST_BASE_A)
            self.delta_mem = float(DELTA_MEM_DECAY_A)
        else:
            self.step = float(STEP_SIZE_B)
            self.vision = float(VISION_RADIUS_B)
            self.alpha_h = float(HEADING_ALPHA_B)
            self.omega0 = float(OMEGA0_B)
            self.persist_base = int(PERSIST_BASE_B)
            self.delta_mem = float(DELTA_MEM_DECAY_B)

        # Resources
        self.energy = float(energy)
        self.hydration = float(hydration)

        self.alive = True
        self.repro_cooldown = 0

        # Memory + exploration + heading (all in (x,y))
        self.mw_x, self.mw_y = 0.0, 0.0
        self.xi_x, self.xi_y = random_unit_vector(self.rng)
        self.persist_count = 0
        self.hx, self.hy = random_unit_vector(self.rng)

        # Social traits (mutable if you want evolution)
        self.theta_same = float(THETA_SAME_DEFAULT)
        self.theta_diff = float(THETA_DIFF_DEFAULT)
        self.theta_mate = float(THETA_MATE_DEFAULT)
        self.theta_rep = float(THETA_AGENT_REPULSION_DEFAULT)
        self.p_mate = float(P_MATE_DEFAULT)

    # ---------- helpers ----------
    def in_nest(self, universe: Universe) -> bool:
        cx, cy = universe.size_x / 2, universe.size_y / 2
        dx = torus_delta(cx, self.x, universe.size_x)
        dy = torus_delta(cy, self.y, universe.size_y)
        return (dx * dx + dy * dy) <= universe.nest_radius ** 2

    def satiety_joint(self) -> float:
        # s(w,f) = sigmoid(k(E-E_sat))*sigmoid(k(H-H_sat))
        Sw = sigmoid(SAT_SIGMOID_K * (self.energy - ENERGY_SAT))
        Sh = sigmoid(SAT_SIGMOID_K * (self.hydration - HYDRATION_SAT))
        return Sw * Sh

    def urgency_weights(self) -> tuple[float, float]:
        Uw = 1.0 / ((self.hydration + EPS) ** URGENCY_P)  # hydration ~ water
        Uf = 1.0 / ((self.energy + EPS) ** URGENCY_P)     # energy ~ food
        denom = Uw + Uf
        if denom <= EPS:
            return 0.5, 0.5
        pi_w = Uw / denom
        return float(pi_w), float(1.0 - pi_w)

    def mate_pref(self, other_type: int) -> float:
        # Independent of basal same/diff attraction.
        if other_type == self.type:
            return self.p_mate
        return 1.0 - self.p_mate

    # ---------- core dynamics ----------
    def _visible_vectors(self, universe: Universe) -> tuple[float, float, float, float]:
        """
        Returns (a_w_vis_x, a_w_vis_y, a_f_vis_x, a_f_vis_y).
        Uses exp kernel with ell = vision/2.
        """
        ell = self.vision / 2.0
        ax_w, ay_w = 0.0, 0.0
        ax_f, ay_f = 0.0, 0.0

        # Water (only 4 sources -> small loop is fine)
        if universe.water_sources.size:
            for wx, wy in universe.water_sources:
                dx = torus_delta(wx, self.x, universe.size_x)
                dy = torus_delta(wy, self.y, universe.size_y)
                d2 = dx * dx + dy * dy
                if d2 <= self.vision * self.vision:
                    d = np.sqrt(d2)
                    w = float(np.exp(-d / (ell + EPS)))
                    ux, uy = unit_vec(dx, dy, EPS_DIR)
                    ax_w += w * ux
                    ay_w += w * uy

        # Food (vectorized)
        if universe.food_x.size:
            dx, dy = torus_delta_vec(universe.food_x, universe.food_y, self.x, self.y, universe.size_x, universe.size_y)
            d2 = dx * dx + dy * dy
            mask = d2 <= (self.vision * self.vision)
            if np.any(mask):
                d = np.sqrt(d2[mask])
                w = exp_kernel(d, ell)
                inv = 1.0 / (d + EPS_DIR)
                ux = dx[mask] * inv
                uy = dy[mask] * inv
                ax_f = float(np.sum(w * ux))
                ay_f = float(np.sum(w * uy))

        return ax_w, ay_w, ax_f, ay_f

    def _update_memory_and_search(self, a_w_vis_x: float, a_w_vis_y: float, pi_w: float) -> tuple[float, float]:
        """
        Computes a_w_eff unit vector (effective direction to water) using:
        - visibility gating L_w
        - memory mw (decay/update)
        - persistent exploration xi (ballistic search)
        """
        vis_norm = safe_norm(a_w_vis_x, a_w_vis_y, EPS_DIR)
        Lw = G0_WATER_SIGNAL / (vis_norm + G0_WATER_SIGNAL)

        # memory update
        if vis_norm > 1e-8:
            uwx, uwy = a_w_vis_x / vis_norm, a_w_vis_y / vis_norm
            self.mw_x = (1.0 - ETA_MEM_UPDATE) * self.mw_x + ETA_MEM_UPDATE * uwx
            self.mw_y = (1.0 - ETA_MEM_UPDATE) * self.mw_y + ETA_MEM_UPDATE * uwy
        else:
            self.mw_x *= (1.0 - self.delta_mem)
            self.mw_y *= (1.0 - self.delta_mem)

        # persistence schedule: more urgent water + no signal -> longer straight runs
        T_persist = int(round(self.persist_base * (1.0 + C_SEEK_PERSIST * pi_w * Lw)))
        T_persist = max(1, min(300, T_persist))

        if self.persist_count >= T_persist:
            self.xi_x, self.xi_y = random_unit_vector(self.rng)
            self.persist_count = 0
        else:
            self.persist_count += 1

        # seek vector combines memory + xi
        seek_x = KAPPA_MEM * self.mw_x + KAPPA_XI * self.xi_x
        seek_y = KAPPA_MEM * self.mw_y + KAPPA_XI * self.xi_y
        seek_x, seek_y = unit_vec(seek_x, seek_y, EPS_DIR)

        # visible unit vector
        vis_u_x, vis_u_y = unit_vec(a_w_vis_x, a_w_vis_y, EPS_DIR)

        # water effective: mix
        eff_x = (1.0 - Lw) * vis_u_x + Lw * seek_x
        eff_y = (1.0 - Lw) * vis_u_y + Lw * seek_y
        eff_x, eff_y = unit_vec(eff_x, eff_y, EPS_DIR)

        return eff_x, eff_y

    def _social_vectors(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> tuple[float, float, float, float, float, float]:
        """
        Returns:
        - g_same_x, g_same_y
        - g_diff_x, g_diff_y
        - g_rep_x,  g_rep_y  (agent-agent repulsion)
        Computed from snapshot arrays (synchronous perception).
        """
        n = xs.size
        if n <= 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        dx, dy = torus_delta_vec(xs, ys, self.x, self.y, universe.size_x, universe.size_y)
        d2 = dx * dx + dy * dy

        # exclude self
        d2[self_idx] = np.inf

        # vision mask
        vis2 = self.vision * self.vision
        mask_vis = d2 <= vis2

        ell = self.vision / 2.0

        # same/diff attraction
        g_same_x = g_same_y = 0.0
        g_diff_x = g_diff_y = 0.0

        if np.any(mask_vis):
            d = np.sqrt(d2[mask_vis])
            w = exp_kernel(d, ell)
            inv = 1.0 / (d + EPS_DIR)
            ux = dx[mask_vis] * inv
            uy = dy[mask_vis] * inv

            t = types[mask_vis]
            same = (t == self.type)
            diff = ~same

            if np.any(same):
                ws = w[same]
                g_same_x = float(np.sum(ws * ux[same]))
                g_same_y = float(np.sum(ws * uy[same]))

            if np.any(diff):
                wd = w[diff]
                g_diff_x = float(np.sum(wd * ux[diff]))
                g_diff_y = float(np.sum(wd * uy[diff]))

        g_same_x *= self.theta_same
        g_same_y *= self.theta_same
        g_diff_x *= self.theta_diff
        g_diff_y *= self.theta_diff

        # short-range repulsion (agent-agent)
        rep2 = AGENT_REPULSION_RADIUS ** 2
        mask_rep = d2 <= rep2
        g_rep_x = g_rep_y = 0.0
        if np.any(mask_rep):
            d = np.sqrt(d2[mask_rep])
            w = gaussian_repulsion(d, AGENT_REPULSION_RADIUS)
            inv = 1.0 / (d + EPS_DIR)
            ux = dx[mask_rep] * inv
            uy = dy[mask_rep] * inv
            # repulsion: push away -> negative of toward-neighbor unit vectors
            g_rep_x = -self.theta_rep * float(np.sum(w * ux))
            g_rep_y = -self.theta_rep * float(np.sum(w * uy))

        return g_same_x, g_same_y, g_diff_x, g_diff_y, g_rep_x, g_rep_y

    def _source_repulsion(self, universe: Universe) -> tuple[float, float]:
        """
        Repel near water source centers to avoid crowding. Must keep r_src < r_drink.
        """
        if not universe.water_sources.size:
            return 0.0, 0.0

        gx = gy = 0.0
        r2 = SOURCE_REPULSION_RADIUS ** 2

        for wx, wy in universe.water_sources:
            dx = torus_delta(wx, self.x, universe.size_x)  # source - self
            dy = torus_delta(wy, self.y, universe.size_y)
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                d = float(np.sqrt(d2))
                w = float(np.exp(- (d / (SOURCE_REPULSION_RADIUS + EPS)) ** 2))
                # push away from source: direction (self - source) = -(source-self)
                ux, uy = unit_vec(-dx, -dy, EPS_DIR)
                gx += w * ux
                gy += w * uy

        return THETA_SOURCE_REPULSION * gx, THETA_SOURCE_REPULSION * gy

    def _mate_vector(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> tuple[float, float]:
        """
        Mate attraction vector: only meaningful under joint satiety (applied later as scaling).
        Uses interaction radius r_e (INTERACTION_RADIUS).
        """
        n = xs.size
        if n <= 1:
            return 0.0, 0.0

        dx, dy = torus_delta_vec(xs, ys, self.x, self.y, universe.size_x, universe.size_y)
        d2 = dx * dx + dy * dy
        d2[self_idx] = np.inf

        re2 = INTERACTION_RADIUS ** 2
        mask = d2 <= re2
        if not np.any(mask):
            return 0.0, 0.0

        d = np.sqrt(d2[mask])
        ell = self.vision / 2.0
        w = exp_kernel(d, ell)

        # preference P_ij
        t = types[mask]
        P = np.where(t == self.type, self.p_mate, 1.0 - self.p_mate)

        inv = 1.0 / (d + EPS_DIR)
        ux = dx[mask] * inv
        uy = dy[mask] * inv

        gx = float(np.sum(w * P * ux))
        gy = float(np.sum(w * P * uy))
        return self.theta_mate * gx, self.theta_mate * gy

    def move(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> None:
        """
        Final movement dynamics:
        - local foraging (water/food) weighted by urgency
        - water "seek" when not visible (memory + ballistic search)
        - basal social same/diff always-on
        - mate attraction scaled by joint satiety (AND)
        - repulsions (agents + water sources)
        - anti-paralysis exploration when signals cancel
        - heading inertia
        - move by step_size (rapidez)
        """
        # (1) visible cues
        a_w_vis_x, a_w_vis_y, a_f_vis_x, a_f_vis_y = self._visible_vectors(universe)

        # (2) urgency weights
        pi_w, pi_f = self.urgency_weights()

        # (3) effective water direction (handles "need water but no water visible")
        a_w_eff_x, a_w_eff_y = self._update_memory_and_search(a_w_vis_x, a_w_vis_y, pi_w)

        # (4) foraging vector
        a_f_u_x, a_f_u_y = unit_vec(a_f_vis_x, a_f_vis_y, EPS_DIR)
        g_for_x = pi_w * a_w_eff_x + pi_f * a_f_u_x
        g_for_y = pi_w * a_w_eff_y + pi_f * a_f_u_y

        # (5) social + repulsions
        g_same_x, g_same_y, g_diff_x, g_diff_y, g_rep_x, g_rep_y = self._social_vectors(
            universe, xs, ys, types, self_idx
        )
        g_src_x, g_src_y = self._source_repulsion(universe)

        # (6) satiety and mate
        sat = self.satiety_joint()
        g_mate_x, g_mate_y = self._mate_vector(universe, xs, ys, types, self_idx)

        # (7) compose (mate competes with foraging)
        g_raw_x = (1.0 - sat) * g_for_x + g_same_x + g_diff_x + sat * g_mate_x + g_rep_x + g_src_x
        g_raw_y = (1.0 - sat) * g_for_y + g_same_y + g_diff_y + sat * g_mate_y + g_rep_y + g_src_y

        # (8) anti-paralysis: add exploration when cancellations happen
        raw_norm = safe_norm(g_raw_x, g_raw_y, EPS_DIR)
        omega = self.omega0 * (G1_CANCEL_SCALE / (raw_norm + G1_CANCEL_SCALE))
        g_tot_x = g_raw_x + omega * self.xi_x
        g_tot_y = g_raw_y + omega * self.xi_y
        g_tot_u_x, g_tot_u_y = unit_vec(g_tot_x, g_tot_y, EPS_DIR)

        # (9) heading inertia
        self.hx = (1.0 - self.alpha_h) * self.hx + self.alpha_h * g_tot_u_x
        self.hy = (1.0 - self.alpha_h) * self.hy + self.alpha_h * g_tot_u_y
        dir_x, dir_y = unit_vec(self.hx, self.hy, EPS_DIR)

        # (10) move by step (rapidez)
        self.x, self.y = wrap_pos(self.x + self.step * dir_x, self.y + self.step * dir_y, universe.size_x, universe.size_y)

    def try_eat(self, universe: Universe) -> bool:
        if not universe.food_x.size:
            return False
        d2 = torus_dist2(universe.food_x, universe.food_y, self.x, self.y, universe.size_x, universe.size_y)
        i = int(np.argmin(d2))
        if d2[i] <= FOOD_EAT_RADIUS ** 2:
            self.energy = min(ENERGY_MAX, self.energy + universe.food_value)
            universe.remove_food_index(i)
            return True
        return False

    def try_drink(self, universe: Universe) -> bool:
        if not universe.water_sources.size:
            return False
        wx = universe.water_sources[:, 0]
        wy = universe.water_sources[:, 1]
        d2 = torus_dist2(wx, wy, self.x, self.y, universe.size_x, universe.size_y)
        if float(np.min(d2)) <= WATER_DRINK_RADIUS ** 2:
            self.hydration = min(HYDRATION_MAX, self.hydration + DRINK_GAIN)
            return True
        return False

    def metabolize(self):
        # Baseline decay
        self.hydration -= BASE_HYDRATION_DECAY
        self.energy -= BASE_ENERGY_DECAY

        # Step cost affects both (your "w and f decrease per step")
        move_cost = MOVE_COST_PER_DISTANCE * self.step
        self.hydration -= move_cost
        self.energy -= move_cost

        # Low hydration penalty on energy
        if self.hydration < LOW_HYDRATION_THRESHOLD:
            self.energy -= LOW_HYDRATION_PENALTY_ENERGY

        # Cooldown
        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

        if self.energy <= 0.0 or self.hydration <= 0.0:
            self.alive = False


# ----------------------------
# Population
# ----------------------------
class Population:
    def __init__(self, cells: list[Cell], universe: Universe):
        self.cells = list(cells)
        self.universe = universe

        self.births = 0
        self.deaths = 0
        self.food_eaten = 0

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array([c.x for c in self.cells if c.alive], dtype=float)
        ys = np.array([c.y for c in self.cells if c.alive], dtype=float)
        ts = np.array([c.type for c in self.cells if c.alive], dtype=int)
        return xs, ys, ts

    def step(self):
        new_cells: list[Cell] = []
        deaths_this_step = 0
        births_this_step = 0
        food_eaten_this_step = 0

        # Synchronous perception snapshot (positions/types at start of tick)
        alive_cells = [c for c in self.cells if c.alive]
        if not alive_cells:
            self.cells = []
            return {
                "alive": 0,
                "births_total": self.births,
                "deaths_total": self.deaths,
                "food_eaten_total": self.food_eaten,
                "births_step": 0,
                "deaths_step": 0,
                "food_eaten_step": 0,
            }

        xs = np.array([c.x for c in alive_cells], dtype=float)
        ys = np.array([c.y for c in alive_cells], dtype=float)
        ts = np.array([c.type for c in alive_cells], dtype=int)

        # Iterate in randomized order to reduce ordering artifacts
        order = np.arange(len(alive_cells))
        np.random.shuffle(order)

        # For reproduction pairing: track who already reproduced this tick
        reproduced = set()

        for local_idx in order:
            c = alive_cells[local_idx]
            if not c.alive:
                continue

            # Move (uses snapshot arrays)
            c.move(self.universe, xs, ys, ts, local_idx)

            # Eat/Drink
            if c.try_eat(self.universe):
                food_eaten_this_step += 1
            c.try_drink(self.universe)

            # Reproduction (optional, partner-based, satiety-gated)
            if ENABLE_REPRODUCTION and c.alive and c.repro_cooldown <= 0 and local_idx not in reproduced:
                child = self._try_reproduce_partner_based(c, alive_cells, xs, ys, ts, local_idx, reproduced)
                if child is not None:
                    new_cells.append(child)
                    births_this_step += 1

            # Metabolize
            c.metabolize()
            if not c.alive:
                deaths_this_step += 1

        # Cull dead, add children
        self.cells = [c for c in self.cells if c.alive] + new_cells

        self.births += births_this_step
        self.deaths += deaths_this_step
        self.food_eaten += food_eaten_this_step

        return {
            "alive": len(self.cells),
            "births_total": self.births,
            "deaths_total": self.deaths,
            "food_eaten_total": self.food_eaten,
            "births_step": births_this_step,
            "deaths_step": deaths_this_step,
            "food_eaten_step": food_eaten_this_step,
        }

    def _try_reproduce_partner_based(
        self,
        c: Cell,
        alive_cells: list[Cell],
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        idx: int,
        reproduced: set[int],
    ) -> Cell | None:
        """
        Partner-based reproduction:
        - requires joint satiety via c.satiety_joint()
        - requires a partner within INTERACTION_RADIUS with cooldown=0 and alive
        - probability scaled by REPRO_BETA * sat_i * sat_j * P_ij
        - costs reduce energy/hydration, which automatically turns mate-mode off after reproduction
        """
        sat_i = c.satiety_joint()
        if sat_i <= 0.05:
            return None

        # Optional: nest requirement
        if REPRO_REQUIRE_NEST and not c.in_nest(self.universe):
            return None

        dx, dy = torus_delta_vec(xs, ys, c.x, c.y, self.universe.size_x, self.universe.size_y)
        d2 = dx * dx + dy * dy
        d2[idx] = np.inf

        re2 = INTERACTION_RADIUS ** 2
        mask = d2 <= re2
        if not np.any(mask):
            return None

        # Candidate indices
        cand_idx = np.where(mask)[0].tolist()
        # filter partners: alive, not in reproduced, cooldown==0, (nest requirement if enabled)
        partners = []
        for j in cand_idx:
            partner = alive_cells[j]
            if not partner.alive:
                continue
            if partner.repro_cooldown > 0:
                continue
            if j in reproduced:
                continue
            if REPRO_REQUIRE_NEST and not partner.in_nest(self.universe):
                continue
            sat_j = partner.satiety_joint()
            if sat_j <= 0.05:
                continue
            partners.append((j, partner, sat_j))

        if not partners:
            return None

        # Choose best partner by preference P_ij (tie-break by closeness)
        def score(item):
            j, partner, sat_j = item
            Pij = c.mate_pref(partner.type)
            return (Pij * sat_j, -d2[j])

        best_j, partner, sat_j = max(partners, key=score)
        Pij = c.mate_pref(partner.type)

        p_rep = REPRO_BETA * sat_i * sat_j * Pij
        if c.rng.uniform(0.0, 1.0) >= p_rep:
            return None

        # Apply costs + cooldown to both
        c.repro_cooldown = REPRO_COOLDOWN_STEPS
        partner.repro_cooldown = REPRO_COOLDOWN_STEPS
        reproduced.add(idx)
        reproduced.add(best_j)

        c.energy = max(0.0, c.energy - REPRO_COST_ENERGY)
        c.hydration = max(0.0, c.hydration - REPRO_COST_HYDRATION)
        partner.energy = max(0.0, partner.energy - REPRO_COST_ENERGY)
        partner.hydration = max(0.0, partner.hydration - REPRO_COST_HYDRATION)

        # Spawn child near c
        ox = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
        oy = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
        child_x, child_y = wrap_pos(c.x + ox, c.y + oy, self.universe.size_x, self.universe.size_y)

        # Child type: simple inheritance (randomly from one parent)
        child_type = c.type if c.rng.uniform(0.0, 1.0) < 0.5 else partner.type

        # Child resources: take share from each parent (optional symmetric transfer)
        child_energy = min(ENERGY_MAX, CHILD_SHARE * (c.energy + partner.energy))
        child_hyd = min(HYDRATION_MAX, CHILD_SHARE * (c.hydration + partner.hydration))

        return Cell(child_x, child_y, child_type, energy=child_energy, hydration=child_hyd, rng=c.rng)

    def positions_energy(self):
        if not self.cells:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        x = np.array([c.x for c in self.cells], dtype=float)
        y = np.array([c.y for c in self.cells], dtype=float)
        e = np.array([c.energy for c in self.cells], dtype=float)
        h = np.array([c.hydration for c in self.cells], dtype=float)
        t = np.array([c.type for c in self.cells], dtype=int)
        return x, y, e, h, t


# ----------------------------
# Simulation + Viz
# ----------------------------
class Simulation:
    def __init__(self, universe: Universe, population: Population):
        self.universe = universe
        self.population = population
        self._ani = None

    def animate(
        self,
        steps: int = 3000,
        food_spawn_per_step: int = FOOD_SPAWN_PER_STEP,
        max_food: int = MAX_FOOD,
        interval_ms: int = 16,
    ):
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=(16, 9), facecolor="black")
        ax.set_facecolor("black")
        ax.set_xlim(0, self.universe.size_x)
        ax.set_ylim(0, self.universe.size_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_alpha(0.25)

        # Star field
        stars = np.random.uniform([0, 0], [self.universe.size_x, self.universe.size_y], size=(350, 2))
        ax.scatter(stars[:, 0], stars[:, 1], s=2, alpha=0.15)

        # Water sources (glow)
        for wx, wy in self.universe.water_sources:
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius, alpha=0.20))
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius * 0.55, alpha=0.28))
            # anti-crowding ring (optional, faint)
            ax.add_patch(plt.Circle((wx, wy), SOURCE_REPULSION_RADIUS, fill=False, linestyle=":", linewidth=0.8, alpha=0.35))

        # Nest ring
        cx, cy = self.universe.size_x / 2, self.universe.size_y / 2
        ax.add_patch(plt.Circle((cx, cy), self.universe.nest_radius, fill=False, linestyle="--", linewidth=1.2, alpha=0.6))

        # Food + cells
        food_sc = ax.scatter([], [], s=18, alpha=0.9)

        # Color by energy; shape/size by energy+hydration
        cell_glow = ax.scatter([], [], s=[], alpha=0.10, cmap="plasma", vmin=PLOT_VMIN_ENERGY, vmax=PLOT_VMAX_ENERGY)
        cell_sc = ax.scatter([], [], s=[], alpha=0.95, cmap="plasma", vmin=PLOT_VMIN_ENERGY, vmax=PLOT_VMAX_ENERGY)

        hud = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            alpha=0.95
        )

        def init():
            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            x, y, e, h, _t = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])
                sizes = 14 + 2.0 * np.sqrt(np.clip(e, 0, 80)) + 1.2 * np.sqrt(np.clip(h, 0, 80))
                cell_glow.set_offsets(xy)
                cell_sc.set_offsets(xy)
                cell_glow.set_sizes(sizes * 3.0)
                cell_sc.set_sizes(sizes)
                cell_glow.set_array(e)
                cell_sc.set_array(e)
            else:
                empty = np.empty((0, 2))
                cell_glow.set_offsets(empty)
                cell_sc.set_offsets(empty)
                cell_glow.set_array(np.array([]))
                cell_sc.set_array(np.array([]))

            hud.set_text(
                f"Step: 0   "
                f"Alive: {len(self.population.cells)}   "
                f"Food: {self.universe.food_x.size}   "
                f"Births: {self.population.births}   "
                f"Deaths: {self.population.deaths}   "
                f"AvgE: {(e.mean() if e.size else 0):.1f}   "
                f"AvgH: {(h.mean() if h.size else 0):.1f}"
            )
            return food_sc, cell_glow, cell_sc, hud

        def update(_frame: int):
            if self.universe.food_x.size < max_food:
                self.universe.spawn_food(food_spawn_per_step)

            stats = self.population.step()

            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            x, y, e, h, _t = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])
                sizes = 14 + 2.0 * np.sqrt(np.clip(e, 0, 80)) + 1.2 * np.sqrt(np.clip(h, 0, 80))
                cell_glow.set_offsets(xy)
                cell_sc.set_offsets(xy)
                cell_glow.set_sizes(sizes * 3.0)
                cell_sc.set_sizes(sizes)
                cell_glow.set_array(e)
                cell_sc.set_array(e)
            else:
                empty = np.empty((0, 2))
                cell_glow.set_offsets(empty)
                cell_sc.set_offsets(empty)
                cell_glow.set_array(np.array([]))
                cell_sc.set_array(np.array([]))

            hud.set_text(
                f"Step: {_frame}   "
                f"Alive: {stats['alive']}   "
                f"Food: {self.universe.food_x.size}   "
                f"Births: +{stats['births_step']} ({stats['births_total']})   "
                f"Deaths: -{stats['deaths_step']} ({stats['deaths_total']})   "
                f"AvgE: {(e.mean() if e.size else 0):.1f}   "
                f"AvgH: {(h.mean() if h.size else 0):.1f}"
            )
            return food_sc, cell_glow, cell_sc, hud

        self._ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval_ms)
        plt.show()


# ----------------------------
# Minimal runner example
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    uni = Universe(rng=rng)
    uni.gen_water_sources()

    # initial food
    uni.spawn_food(MAX_FOOD // 2)

    # initial population: mix of two types
    cells = []
    for _ in range(35):
        x = rng.uniform(0, uni.size_x)
        y = rng.uniform(0, uni.size_y)
        cells.append(Cell(x, y, cell_type=TYPE_A, energy=18.0, hydration=12.0, rng=rng))
    for _ in range(35):
        x = rng.uniform(0, uni.size_x)
        y = rng.uniform(0, uni.size_y)
        cells.append(Cell(x, y, cell_type=TYPE_B, energy=18.0, hydration=12.0, rng=rng))

    pop = Population(cells, uni)
    sim = Simulation(uni, pop)
    sim.animate(steps=4000, food_spawn_per_step=FOOD_SPAWN_PER_STEP, max_food=MAX_FOOD, interval_ms=16)