from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Params import *
from Utilities import *


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
        Even-ish distribution with separation constraints (torus-aware).
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

        # fallback: quadrant placement
        if len(positions) < n_sources:
            positions = [
                (self.size_x * 0.25, self.size_y * 0.25),
                (self.size_x * 0.25, self.size_y * 0.75),
                (self.size_x * 0.75, self.size_y * 0.25),
                (self.size_x * 0.75, self.size_y * 0.75),
            ][:n_sources]

        self.water_sources = np.array(positions, dtype=float)

    def spawn_food(self, n_new: int):
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

            # avoid water blobs (visual radius)
            if self.water_sources.size:
                d2w = torus_dist2(self.water_sources[:, 0], self.water_sources[:, 1], x, y, self.size_x, self.size_y)
                if np.any(d2w < (self.water_sources_radius ** 2)):
                    attempts += 1
                    continue

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

        self.energy = float(energy)
        self.hydration = float(hydration)

        self.alive = True
        self.repro_cooldown = 0

        # Memory + exploration + heading
        self.mw_x, self.mw_y = 0.0, 0.0
        self.xi_x, self.xi_y = random_unit_vector(self.rng)
        self.persist_count = 0
        self.hx, self.hy = random_unit_vector(self.rng)

        # Social traits
        self.theta_same = float(THETA_SAME_DEFAULT)
        self.theta_diff = float(THETA_DIFF_DEFAULT)
        self.theta_mate = float(THETA_MATE_DEFAULT)
        self.theta_rep = float(THETA_AGENT_REPULSION_DEFAULT)
        self.p_mate = float(P_MATE_DEFAULT)

    def in_nest(self, universe: Universe) -> bool:
        cx, cy = universe.size_x / 2, universe.size_y / 2
        dx = torus_delta(cx, self.x, universe.size_x)
        dy = torus_delta(cy, self.y, universe.size_y)
        return (dx * dx + dy * dy) <= universe.nest_radius ** 2

    def satiety_joint(self) -> float:
        Sw = sigmoid(SAT_SIGMOID_K * (self.energy - ENERGY_SAT))
        Sh = sigmoid(SAT_SIGMOID_K * (self.hydration - HYDRATION_SAT))
        return Sw * Sh

    def urgency_weights(self) -> tuple[float, float]:
        Uw = 1.0 / ((self.hydration + EPS) ** URGENCY_P)
        Uf = 1.0 / ((self.energy + EPS) ** URGENCY_P)
        denom = Uw + Uf
        if denom <= EPS:
            return 0.5, 0.5
        pi_w = Uw / denom
        return float(pi_w), float(1.0 - pi_w)

    def mate_pref(self, other_type: int) -> float:
        return self.p_mate if other_type == self.type else (1.0 - self.p_mate)

    def _visible_vectors(self, universe: Universe) -> tuple[float, float, float, float]:
        ell = self.vision / 2.0
        ax_w, ay_w = 0.0, 0.0
        ax_f, ay_f = 0.0, 0.0

        if universe.water_sources.size:
            for wx, wy in universe.water_sources:
                dx = torus_delta(wx, self.x, universe.size_x)
                dy = torus_delta(wy, self.y, universe.size_y)
                d2 = dx * dx + dy * dy
                if d2 <= self.vision * self.vision:
                    d = float(np.sqrt(d2))
                    w = float(np.exp(-d / (ell + EPS)))
                    ux, uy = unit_vec(dx, dy, EPS_DIR)
                    ax_w += w * ux
                    ay_w += w * uy

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
        vis_norm = safe_norm(a_w_vis_x, a_w_vis_y, EPS_DIR)
        Lw = G0_WATER_SIGNAL / (vis_norm + G0_WATER_SIGNAL)

        if vis_norm > 1e-8:
            uwx, uwy = a_w_vis_x / vis_norm, a_w_vis_y / vis_norm
            self.mw_x = (1.0 - ETA_MEM_UPDATE) * self.mw_x + ETA_MEM_UPDATE * uwx
            self.mw_y = (1.0 - ETA_MEM_UPDATE) * self.mw_y + ETA_MEM_UPDATE * uwy
        else:
            self.mw_x *= (1.0 - self.delta_mem)
            self.mw_y *= (1.0 - self.delta_mem)

        T_persist = int(round(self.persist_base * (1.0 + C_SEEK_PERSIST * pi_w * Lw)))
        T_persist = max(1, min(300, T_persist))

        if self.persist_count >= T_persist:
            self.xi_x, self.xi_y = random_unit_vector(self.rng)
            self.persist_count = 0
        else:
            self.persist_count += 1

        seek_x = KAPPA_MEM * self.mw_x + KAPPA_XI * self.xi_x
        seek_y = KAPPA_MEM * self.mw_y + KAPPA_XI * self.xi_y
        seek_x, seek_y = unit_vec(seek_x, seek_y, EPS_DIR)

        vis_u_x, vis_u_y = unit_vec(a_w_vis_x, a_w_vis_y, EPS_DIR)

        eff_x = (1.0 - Lw) * vis_u_x + Lw * seek_x
        eff_y = (1.0 - Lw) * vis_u_y + Lw * seek_y
        eff_x, eff_y = unit_vec(eff_x, eff_y, EPS_DIR)

        return eff_x, eff_y

    def _social_vectors(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> tuple[float, float, float, float, float, float]:
        n = xs.size
        if n <= 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        dx, dy = torus_delta_vec(xs, ys, self.x, self.y, universe.size_x, universe.size_y)
        d2 = dx * dx + dy * dy
        d2[self_idx] = np.inf

        vis2 = self.vision * self.vision
        mask_vis = d2 <= vis2
        ell = self.vision / 2.0

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

        rep2 = AGENT_REPULSION_RADIUS ** 2
        mask_rep = d2 <= rep2
        g_rep_x = g_rep_y = 0.0
        if np.any(mask_rep):
            d = np.sqrt(d2[mask_rep])
            w = gaussian_repulsion(d, AGENT_REPULSION_RADIUS)
            inv = 1.0 / (d + EPS_DIR)
            ux = dx[mask_rep] * inv
            uy = dy[mask_rep] * inv
            g_rep_x = -self.theta_rep * float(np.sum(w * ux))
            g_rep_y = -self.theta_rep * float(np.sum(w * uy))

        return g_same_x, g_same_y, g_diff_x, g_diff_y, g_rep_x, g_rep_y

    def _source_repulsion(self, universe: Universe) -> tuple[float, float]:
        if not universe.water_sources.size:
            return 0.0, 0.0

        gx = gy = 0.0
        r2 = SOURCE_REPULSION_RADIUS ** 2

        for wx, wy in universe.water_sources:
            dx = torus_delta(wx, self.x, universe.size_x)
            dy = torus_delta(wy, self.y, universe.size_y)
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                d = float(np.sqrt(d2))
                w = float(np.exp(- (d / (SOURCE_REPULSION_RADIUS + EPS)) ** 2))
                ux, uy = unit_vec(-dx, -dy, EPS_DIR)  # away from source
                gx += w * ux
                gy += w * uy

        return THETA_SOURCE_REPULSION * gx, THETA_SOURCE_REPULSION * gy

    def _mate_vector(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> tuple[float, float]:
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

        t = types[mask]
        P = np.where(t == self.type, self.p_mate, 1.0 - self.p_mate)

        inv = 1.0 / (d + EPS_DIR)
        ux = dx[mask] * inv
        uy = dy[mask] * inv

        gx = float(np.sum(w * P * ux))
        gy = float(np.sum(w * P * uy))
        return self.theta_mate * gx, self.theta_mate * gy

    def move(self, universe: Universe, xs: np.ndarray, ys: np.ndarray, types: np.ndarray, self_idx: int) -> None:
        a_w_vis_x, a_w_vis_y, a_f_vis_x, a_f_vis_y = self._visible_vectors(universe)

        pi_w, pi_f = self.urgency_weights()
        a_w_eff_x, a_w_eff_y = self._update_memory_and_search(a_w_vis_x, a_w_vis_y, pi_w)

        a_f_u_x, a_f_u_y = unit_vec(a_f_vis_x, a_f_vis_y, EPS_DIR)
        g_for_x = pi_w * a_w_eff_x + pi_f * a_f_u_x
        g_for_y = pi_w * a_w_eff_y + pi_f * a_f_u_y

        g_same_x, g_same_y, g_diff_x, g_diff_y, g_rep_x, g_rep_y = self._social_vectors(
            universe, xs, ys, types, self_idx
        )
        g_src_x, g_src_y = self._source_repulsion(universe)

        sat = self.satiety_joint()
        g_mate_x, g_mate_y = self._mate_vector(universe, xs, ys, types, self_idx)

        g_raw_x = (1.0 - sat) * g_for_x + g_same_x + g_diff_x + sat * g_mate_x + g_rep_x + g_src_x
        g_raw_y = (1.0 - sat) * g_for_y + g_same_y + g_diff_y + sat * g_mate_y + g_rep_y + g_src_y

        raw_norm = safe_norm(g_raw_x, g_raw_y, EPS_DIR)
        omega = self.omega0 * (G1_CANCEL_SCALE / (raw_norm + G1_CANCEL_SCALE))
        g_tot_x = g_raw_x + omega * self.xi_x
        g_tot_y = g_raw_y + omega * self.xi_y
        g_tot_u_x, g_tot_u_y = unit_vec(g_tot_x, g_tot_y, EPS_DIR)

        self.hx = (1.0 - self.alpha_h) * self.hx + self.alpha_h * g_tot_u_x
        self.hy = (1.0 - self.alpha_h) * self.hy + self.alpha_h * g_tot_u_y
        dir_x, dir_y = unit_vec(self.hx, self.hy, EPS_DIR)

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
        self.hydration -= BASE_HYDRATION_DECAY
        self.energy -= BASE_ENERGY_DECAY

        move_cost = MOVE_COST_PER_DISTANCE * self.step
        self.hydration -= move_cost
        self.energy -= move_cost

        if self.hydration < LOW_HYDRATION_THRESHOLD:
            self.energy -= LOW_HYDRATION_PENALTY_ENERGY

        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

        if self.energy <= 0.0 or self.hydration <= 0.0:
            self.alive = False


# ----------------------------
# Population (stats por tipo)
# ----------------------------
class Population:
    def __init__(self, cells: list[Cell], universe: Universe):
        self.cells = list(cells)
        self.universe = universe

        self.births_total = 0
        self.deaths_total = 0
        self.food_eaten_total = 0

        self.births_by_type = {TYPE_A: 0, TYPE_B: 0}
        self.deaths_by_type = {TYPE_A: 0, TYPE_B: 0}

        # Para mortalidad acumulada por tipo: "creados" = inicial + nacidos
        self.initial_by_type = {TYPE_A: 0, TYPE_B: 0}
        for c in self.cells:
            if c.alive:
                self.initial_by_type[c.type] += 1

    def step(self):
        new_cells: list[Cell] = []

        births_step = 0
        deaths_step = 0
        food_eaten_step = 0

        births_step_by_type = {TYPE_A: 0, TYPE_B: 0}
        deaths_step_by_type = {TYPE_A: 0, TYPE_B: 0}

        alive_cells = [c for c in self.cells if c.alive]
        if not alive_cells:
            self.cells = []
            return self._stats_dict(
                step_alive=0,
                nA0=0, nB0=0,
                births_step=0, deaths_step=0, food_eaten_step=0,
                births_step_by_type=births_step_by_type,
                deaths_step_by_type=deaths_step_by_type,
            )

        # Conteo al inicio del tick (para tasas por step)
        ts0 = np.array([c.type for c in alive_cells], dtype=int)
        nA0 = int(np.sum(ts0 == TYPE_A))
        nB0 = int(np.sum(ts0 == TYPE_B))

        # Snapshot sincrónico para percepción de movimiento
        xs0 = np.array([c.x for c in alive_cells], dtype=float)
        ys0 = np.array([c.y for c in alive_cells], dtype=float)
        ts0 = np.array([c.type for c in alive_cells], dtype=int)

        # (1) MOVER a todos usando snapshot (orden indiferente)
        for idx, c in enumerate(alive_cells):
            if c.alive:
                c.move(self.universe, xs0, ys0, ts0, idx)

        # (2) RESOLVER COLISIONES (no-overlap) después del movimiento
        self._resolve_collisions(alive_cells)

        # (3) Snapshot post-colisión (para reproducción por cercanía, etc.)
        xs = np.array([c.x for c in alive_cells], dtype=float)
        ys = np.array([c.y for c in alive_cells], dtype=float)
        ts = np.array([c.type for c in alive_cells], dtype=int)

        # (4) Interacciones en orden aleatorio
        order = np.arange(len(alive_cells))
        self.universe.rng.shuffle(order)

        reproduced: set[int] = set()

        for local_idx in order:
            c = alive_cells[local_idx]
            if not c.alive:
                continue

            # Eat/Drink con posición ya “limpia” (sin overlap)
            if c.try_eat(self.universe):
                food_eaten_step += 1
            c.try_drink(self.universe)

            # Reproducción usa arrays post-colisión
            if ENABLE_REPRODUCTION and c.repro_cooldown <= 0 and local_idx not in reproduced:
                child = self._try_reproduce_partner_based(
                    c, alive_cells, xs, ys, ts, local_idx, reproduced
                )
                if child is not None:
                    new_cells.append(child)
                    births_step += 1
                    births_step_by_type[child.type] += 1

            # Metabolismo y muerte
            c.metabolize()
            if not c.alive:
                deaths_step += 1
                deaths_step_by_type[c.type] += 1

        # (5) Culling + agregar niños
        survivors = [c for c in alive_cells if c.alive]
        self.cells = survivors + new_cells

        # (6) Opcional pero recomendable: colisión inmediata incluyendo niños
        if self.cells:
            self._resolve_collisions(self.cells)

        # Update totals
        self.births_total += births_step
        self.deaths_total += deaths_step
        self.food_eaten_total += food_eaten_step

        self.births_by_type[TYPE_A] += births_step_by_type[TYPE_A]
        self.births_by_type[TYPE_B] += births_step_by_type[TYPE_B]
        self.deaths_by_type[TYPE_A] += deaths_step_by_type[TYPE_A]
        self.deaths_by_type[TYPE_B] += deaths_step_by_type[TYPE_B]

        return self._stats_dict(
            step_alive=len(self.cells),
            nA0=nA0, nB0=nB0,
            births_step=births_step,
            deaths_step=deaths_step,
            food_eaten_step=food_eaten_step,
            births_step_by_type=births_step_by_type,
            deaths_step_by_type=deaths_step_by_type,
        )

    def _stats_dict(
        self,
        step_alive: int,
        nA0: int, nB0: int,
        births_step: int,
        deaths_step: int,
        food_eaten_step: int,
        births_step_by_type: dict[int, int],
        deaths_step_by_type: dict[int, int],
    ) -> dict:
        # Conteos actuales (post-step)
        x, y, e, h, t = self.positions_energy()
        nA = int(np.sum(t == TYPE_A)) if t.size else 0
        nB = int(np.sum(t == TYPE_B)) if t.size else 0
        n = int(step_alive)

        # Tasas por step (normalizadas por población al inicio del tick)
        birth_rate = safe_div(births_step, (nA0 + nB0), 0.0)
        mortA_rate = safe_div(deaths_step_by_type[TYPE_A], nA0, 0.0)
        mortB_rate = safe_div(deaths_step_by_type[TYPE_B], nB0, 0.0)

        # Mortalidad acumulada por tipo (muertos / creados)
        createdA = self.initial_by_type[TYPE_A] + self.births_by_type[TYPE_A]
        createdB = self.initial_by_type[TYPE_B] + self.births_by_type[TYPE_B]
        cum_mortA = safe_div(self.deaths_by_type[TYPE_A], createdA, 0.0)
        cum_mortB = safe_div(self.deaths_by_type[TYPE_B], createdB, 0.0)

        # Proporciones actuales
        propA = safe_div(nA, n, 0.0)
        propB = safe_div(nB, n, 0.0)

        return {
            "alive": n,
            "alive_A": nA,
            "alive_B": nB,
            "prop_A": propA,
            "prop_B": propB,

            "births_step": births_step,
            "births_step_A": births_step_by_type[TYPE_A],
            "births_step_B": births_step_by_type[TYPE_B],
            "births_total": self.births_total,
            "births_total_A": self.births_by_type[TYPE_A],
            "births_total_B": self.births_by_type[TYPE_B],
            "birth_rate_step": birth_rate,

            "deaths_step": deaths_step,
            "deaths_step_A": deaths_step_by_type[TYPE_A],
            "deaths_step_B": deaths_step_by_type[TYPE_B],
            "deaths_total": self.deaths_total,
            "deaths_total_A": self.deaths_by_type[TYPE_A],
            "deaths_total_B": self.deaths_by_type[TYPE_B],
            "mort_rate_step_A": mortA_rate,
            "mort_rate_step_B": mortB_rate,
            "cum_mort_A": cum_mortA,
            "cum_mort_B": cum_mortB,

            "food_eaten_step": food_eaten_step,
            "food_eaten_total": self.food_eaten_total,
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
        sat_i = c.satiety_joint()
        if sat_i <= 0.05:
            return None

        if REPRO_REQUIRE_NEST and not c.in_nest(self.universe):
            return None

        dx, dy = torus_delta_vec(xs, ys, c.x, c.y, self.universe.size_x, self.universe.size_y)
        d2 = dx * dx + dy * dy
        d2[idx] = np.inf

        re2 = INTERACTION_RADIUS ** 2
        mask = d2 <= re2
        if not np.any(mask):
            return None

        cand_idx = np.where(mask)[0].tolist()
        partners = []
        for j in cand_idx:
            partner = alive_cells[j]
            if (not partner.alive) or (partner.repro_cooldown > 0) or (j in reproduced):
                continue
            if REPRO_REQUIRE_NEST and not partner.in_nest(self.universe):
                continue
            sat_j = partner.satiety_joint()
            if sat_j <= 0.05:
                continue
            partners.append((j, partner, sat_j))

        if not partners:
            return None

        def score(item):
            j, partner, sat_j = item
            Pij = c.mate_pref(partner.type)
            return (Pij * sat_j, -d2[j])

        best_j, partner, sat_j = max(partners, key=score)
        Pij = c.mate_pref(partner.type)

        p_rep = REPRO_BETA * sat_i * sat_j * Pij
        if c.rng.uniform(0.0, 1.0) >= p_rep:
            return None

        c.repro_cooldown = REPRO_COOLDOWN_STEPS
        partner.repro_cooldown = REPRO_COOLDOWN_STEPS
        reproduced.add(idx)
        reproduced.add(best_j)

        c.energy = max(0.0, c.energy - REPRO_COST_ENERGY)
        c.hydration = max(0.0, c.hydration - REPRO_COST_HYDRATION)
        partner.energy = max(0.0, partner.energy - REPRO_COST_ENERGY)
        partner.hydration = max(0.0, partner.hydration - REPRO_COST_HYDRATION)

        ox = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
        oy = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
        child_x, child_y = wrap_pos(c.x + ox, c.y + oy, self.universe.size_x, self.universe.size_y)

        child_type = c.type if c.rng.uniform(0.0, 1.0) < 0.5 else partner.type

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
    
    def _resolve_collisions(self, agents: list[Cell]) -> None:
        """
        Hard no-overlap solver (PBD) on a torus.
        Enforces ||x_i - x_j|| >= R_i + R_j via iterative position corrections.
        """
        n = len(agents)
        if n <= 1:
            return

        Lx, Ly = self.universe.size_x, self.universe.size_y

        xs = np.array([c.x for c in agents], dtype=float)
        ys = np.array([c.y for c in agents], dtype=float)

        # Radio por agente (puedes hacerlo dependiente del tipo si quieres)
        rs = np.full(n, AGENT_COLLISION_RADIUS, dtype=float)

        for _ in range(COLLISION_ITERS):
            for i in range(n - 1):
                for j in range(i + 1, n):
                    dx = torus_delta(xs[j], xs[i], Lx)  # i -> j (mínimo en toro)
                    dy = torus_delta(ys[j], ys[i], Ly)
                    min_d = rs[i] + rs[j]
                    min_d2 = min_d * min_d
                    d2 = dx * dx + dy * dy

                    if d2 >= min_d2:
                        continue

                    # Si están exactamente encima, elige dirección aleatoria
                    if d2 <= COLLISION_EPS:
                        ux, uy = random_unit_vector(self.universe.rng)
                        d = 0.0
                    else:
                        d = math.sqrt(d2)
                        inv = 1.0 / (d + COLLISION_EPS)
                        ux = dx * inv
                        uy = dy * inv

                    penetration = min_d - d
                    if penetration <= 0:
                        continue

                    corr = 0.5 * COLLISION_DAMPING * penetration

                    # Empuja a i y j en direcciones opuestas
                    xs[i] -= corr * ux
                    ys[i] -= corr * uy
                    xs[j] += corr * ux
                    ys[j] += corr * uy

                    xs[i], ys[i] = wrap_pos(xs[i], ys[i], Lx, Ly)
                    xs[j], ys[j] = wrap_pos(xs[j], ys[j], Lx, Ly)

        # escribe de vuelta
        for k, c in enumerate(agents):
            c.x = float(xs[k])
            c.y = float(ys[k])


# ----------------------------
# Simulation + Viz (colores por tipo + HUD con stats)
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
        stars = self.universe.rng.uniform([0, 0], [self.universe.size_x, self.universe.size_y], size=(350, 2))
        ax.scatter(stars[:, 0], stars[:, 1], s=2, alpha=0.15)

        # Water sources
        for wx, wy in self.universe.water_sources:
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius, alpha=0.20))
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius * 0.55, alpha=0.28))
            ax.add_patch(plt.Circle((wx, wy), SOURCE_REPULSION_RADIUS, fill=False, linestyle=":", linewidth=0.8, alpha=0.35))

        # Nest ring
        cx, cy = self.universe.size_x / 2, self.universe.size_y / 2
        ax.add_patch(plt.Circle((cx, cy), self.universe.nest_radius, fill=False, linestyle="--", linewidth=1.2, alpha=0.6))

        # Food scatter (fijo)
        food_sc = ax.scatter([], [], s=18, alpha=0.9, c=FOOD_COLOR)

        # Dos tipos: 2 capas por tipo (glow + main)
        glow_A = ax.scatter([], [], s=[], alpha=0.10, c=TYPE_A_COLOR)
        main_A = ax.scatter([], [], s=[], alpha=0.95, c=TYPE_A_COLOR)

        glow_B = ax.scatter([], [], s=[], alpha=0.10, c=TYPE_B_COLOR)
        main_B = ax.scatter([], [], s=[], alpha=0.95, c=TYPE_B_COLOR)

        # Mini-leyenda textual (evita legend() que suele ensuciar con animación)
        legend = ax.text(
            0.02, 0.06,
            f"{TYPE_A_NAME}: {TYPE_A_COLOR}   |   {TYPE_B_NAME}: {TYPE_B_COLOR}",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9,
            alpha=0.8
        )

        hud = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=HUD_FONT_SIZE,
            alpha=0.95
        )

        def _sizes(e: np.ndarray, h: np.ndarray) -> np.ndarray:
            return (
                CELL_SIZE_BASE
                + CELL_SIZE_ENERGY_SCALE * np.sqrt(np.clip(e, 0, ENERGY_MAX))
                + CELL_SIZE_HYDR_SCALE * np.sqrt(np.clip(h, 0, HYDRATION_MAX))
            )

        def init():
            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            x, y, e, h, t = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])
                sizes = _sizes(e, h)

                mA = (t == TYPE_A)
                mB = (t == TYPE_B)

                glow_A.set_offsets(xy[mA])
                main_A.set_offsets(xy[mA])
                glow_A.set_sizes(sizes[mA] * GLOW_MULT)
                main_A.set_sizes(sizes[mA])

                glow_B.set_offsets(xy[mB])
                main_B.set_offsets(xy[mB])
                glow_B.set_sizes(sizes[mB] * GLOW_MULT)
                main_B.set_sizes(sizes[mB])
            else:
                empty = np.empty((0, 2))
                for sc in (glow_A, main_A, glow_B, main_B):
                    sc.set_offsets(empty)
                    sc.set_sizes([])

            # stats iniciales (sin step)
            n = len(self.population.cells)
            nA = int(np.sum(t == TYPE_A)) if t.size else 0
            nB = int(np.sum(t == TYPE_B)) if t.size else 0
            propA = safe_div(nA, n, 0.0)
            propB = safe_div(nB, n, 0.0)

            hud.set_text(
                "Step: 0\n"
                f"Alive: {n}  |  A: {nA} ({fmt_pct(propA)})  B: {nB} ({fmt_pct(propB)})  |  Food: {self.universe.food_x.size}\n"
                f"Births: 0 (rate 0.0%)  |  Deaths A: 0  B: 0\n"
                f"CumMort A: 0.0%  B: 0.0%  |  AvgE: {(e.mean() if e.size else 0):.1f}  AvgH: {(h.mean() if h.size else 0):.1f}"
            )
            return food_sc, glow_A, main_A, glow_B, main_B, hud, legend

        def update(frame: int):
            if self.universe.food_x.size < max_food:
                self.universe.spawn_food(food_spawn_per_step)

            stats = self.population.step()

            # Food
            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            # Cells by type
            x, y, e, h, t = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])
                sizes = _sizes(e, h)

                mA = (t == TYPE_A)
                mB = (t == TYPE_B)

                glow_A.set_offsets(xy[mA])
                main_A.set_offsets(xy[mA])
                glow_A.set_sizes(sizes[mA] * GLOW_MULT)
                main_A.set_sizes(sizes[mA])

                glow_B.set_offsets(xy[mB])
                main_B.set_offsets(xy[mB])
                glow_B.set_sizes(sizes[mB] * GLOW_MULT)
                main_B.set_sizes(sizes[mB])
            else:
                empty = np.empty((0, 2))
                for sc in (glow_A, main_A, glow_B, main_B):
                    sc.set_offsets(empty)
                    sc.set_sizes([])

            # HUD stats solicitadas
            hud.set_text(
                f"Step: {frame}\n"
                f"Alive: {stats['alive']}  |  A: {stats['alive_A']} ({fmt_pct(stats['prop_A'])})  "
                f"B: {stats['alive_B']} ({fmt_pct(stats['prop_B'])})  |  Food: {self.universe.food_x.size}\n"
                f"Births: +{stats['births_step']} (A +{stats['births_step_A']}, B +{stats['births_step_B']})  "
                f"rate {fmt_pct(stats['birth_rate_step'])}  |  Total {stats['births_total']}\n"
                f"Deaths: -{stats['deaths_step']} (A -{stats['deaths_step_A']}, B -{stats['deaths_step_B']})  "
                f"mortA {fmt_pct(stats['mort_rate_step_A'])}  mortB {fmt_pct(stats['mort_rate_step_B'])}  |  Total {stats['deaths_total']}\n"
                f"CumMort A: {fmt_pct(stats['cum_mort_A'])}  B: {fmt_pct(stats['cum_mort_B'])}  "
                f"|  AvgE: {(e.mean() if e.size else 0):.1f}  AvgH: {(h.mean() if h.size else 0):.1f}"
            )
            return food_sc, glow_A, main_A, glow_B, main_B, hud, legend

        self._ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval_ms)
        plt.show()


# ----------------------------
# Minimal runner
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    uni = Universe(rng=rng)
    uni.gen_water_sources()

    uni.spawn_food(MAX_FOOD // 2)

    cells: list[Cell] = []
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
