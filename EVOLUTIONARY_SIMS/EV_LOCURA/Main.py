# main_sim.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from Params import *
from Utilities import *

# ----------------------------
# World
# ----------------------------
class Universe:
    def __init__(self, rng: np.random.Generator | None = None):
        self.size_x = float(GRID_SIZE_X)
        self.size_y = float(GRID_SIZE_Y)
        self.rng = rng or np.random.default_rng()

        self.food_x = np.empty(0, dtype=float)
        self.food_y = np.empty(0, dtype=float)
        self.water_sources = np.empty((0, 2), dtype=float)

    def gen_water_sources(self, n_sources: int = N_WATER_SOURCES):
        positions: list[tuple[float, float]] = []
        attempts, max_attempts = 0, 6000
        min_sep2 = (WATER_SOURCES_RADIUS * 2.5) ** 2

        while len(positions) < n_sources and attempts < max_attempts:
            x = self.rng.uniform(0, self.size_x)
            y = self.rng.uniform(0, self.size_y)

            ok = True
            for wx, wy in positions:
                dx = torus_delta(x, wx, self.size_x)
                dy = torus_delta(y, wy, self.size_y)
                if (dx * dx + dy * dy) < min_sep2:
                    ok = False
                    break

            if ok:
                positions.append((x, y))
            attempts += 1

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

        min_sep2 = FOOD_MIN_SEP ** 2

        while len(fx) < n_new and attempts < max_attempts:
            x = self.rng.uniform(0, self.size_x)
            y = self.rng.uniform(0, self.size_y)

            # no food inside water blobs (visual radius)
            if self.water_sources.size:
                d2w = torus_dist2(self.water_sources[:, 0], self.water_sources[:, 1], x, y, self.size_x, self.size_y)
                if np.any(d2w < (WATER_SOURCES_RADIUS ** 2)):
                    attempts += 1
                    continue

            # soft separation with existing and new
            ok = True
            if self.food_x.size:
                d2e = torus_dist2(self.food_x, self.food_y, x, y, self.size_x, self.size_y)
                if np.any(d2e < min_sep2):
                    ok = False
            if ok and fx:
                d2n = torus_dist2(np.array(fx), np.array(fy), x, y, self.size_x, self.size_y)
                if np.any(d2n < min_sep2):
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
    def __init__(self, x: float, y: float, cell_type: int, w: float, f: float, rng: np.random.Generator):
        self.x = float(x)
        self.y = float(y)
        self.type = int(cell_type)
        self.rng = rng

        # stocks
        self.w = float(w)  # water stock
        self.f = float(f)  # food stock

        # type params
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

        # evolving traits (start at 0)
        self.theta_same = float(THETA_SAME_INIT)
        self.theta_diff = float(THETA_DIFF_INIT)

        # state
        self.alive = True
        self.repro_cooldown = 0

        # memory/search heading
        self.mw_x, self.mw_y = 0.0, 0.0
        self.xi_x, self.xi_y = random_unit_vector(self.rng)
        self.persist_count = 0
        self.hx, self.hy = random_unit_vector(self.rng)

    @property
    def E(self) -> float:
        return E_of(self.w, self.f)

    def satiety_joint(self) -> float:
        Sw = sigmoid(SAT_SIGMOID_K * (self.w - W_SAT))
        Sf = sigmoid(SAT_SIGMOID_K * (self.f - F_SAT))
        return Sw * Sf

    def urgency_weights(self) -> tuple[float, float]:
        # returns (pi_w, pi_f)
        if URGENCY_USE_MARGINALS:
            mw = dE_dw(self.w, self.f)
            mf = dE_df(self.w, self.f)
            s = mw + mf
            if s <= EPS:
                return 0.5, 0.5
            return float(mw / s), float(mf / s)

        Uw = 1.0 / ((self.w + EPS) ** URGENCY_P)
        Uf = 1.0 / ((self.f + EPS) ** URGENCY_P)
        s = Uw + Uf
        if s <= EPS:
            return 0.5, 0.5
        return float(Uw / s), float(Uf / s)

    # ----- perception -----
    def _visible_vectors(self, uni: Universe) -> tuple[float, float, float, float]:
        ell = self.vision / 2.0
        ax_w = ay_w = 0.0
        ax_f = ay_f = 0.0

        # water sources: small loop
        if uni.water_sources.size:
            for wx, wy in uni.water_sources:
                dx = torus_delta(wx, self.x, uni.size_x)
                dy = torus_delta(wy, self.y, uni.size_y)
                d2 = dx * dx + dy * dy
                if d2 <= self.vision * self.vision:
                    d = float(np.sqrt(d2))
                    wgt = float(np.exp(-d / (ell + EPS)))
                    ux, uy = unit_vec(dx, dy, EPS_DIR)
                    ax_w += wgt * ux
                    ay_w += wgt * uy

        # food: vectorized
        if uni.food_x.size:
            dx, dy = torus_delta_vec(uni.food_x, uni.food_y, self.x, self.y, uni.size_x, uni.size_y)
            d2 = dx * dx + dy * dy
            mask = d2 <= (self.vision * self.vision)
            if np.any(mask):
                d = np.sqrt(d2[mask])
                wgt = exp_kernel(d, ell)
                inv = 1.0 / (d + EPS_DIR)
                ux = dx[mask] * inv
                uy = dy[mask] * inv
                ax_f = float(np.sum(wgt * ux))
                ay_f = float(np.sum(wgt * uy))

        return ax_w, ay_w, ax_f, ay_f

    def _update_water_seek(self, a_w_vis_x: float, a_w_vis_y: float, pi_w: float) -> tuple[float, float]:
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

        # persistence schedule
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
        return unit_vec(eff_x, eff_y, EPS_DIR)

    def _social_vectors(self, uni: Universe, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, self_idx: int) -> tuple[float, float, float, float]:
        """
        Basal same/diff influence only in a ring: [AGENT_REPULSION_RADIUS, D_THETA]
        so it doesn't dominate micro-dynamics when too close.
        """
        n = xs.size
        if n <= 1:
            return 0.0, 0.0, 0.0, 0.0

        dx, dy = torus_delta_vec(xs, ys, self.x, self.y, uni.size_x, uni.size_y)
        d2 = dx * dx + dy * dy
        d2[self_idx] = np.inf

        vis2 = self.vision * self.vision
        mask_vis = d2 <= vis2
        if not np.any(mask_vis):
            return 0.0, 0.0, 0.0, 0.0

        d = np.sqrt(d2[mask_vis])
        # ring window
        mask_ring = (d >= AGENT_REPULSION_RADIUS) & (d <= D_THETA)
        if not np.any(mask_ring):
            return 0.0, 0.0, 0.0, 0.0

        d = d[mask_ring]
        ell = self.vision / 2.0
        wgt = exp_kernel(d, ell)

        idxs = np.where(mask_vis)[0][mask_ring]
        inv = 1.0 / (d + EPS_DIR)
        ux = dx[idxs] * inv
        uy = dy[idxs] * inv

        same = (ts[idxs] == self.type)
        diff = ~same

        g_same_x = g_same_y = 0.0
        g_diff_x = g_diff_y = 0.0

        if np.any(same):
            ws = wgt[same]
            g_same_x = float(np.sum(ws * ux[same]))
            g_same_y = float(np.sum(ws * uy[same]))
        if np.any(diff):
            wd = wgt[diff]
            g_diff_x = float(np.sum(wd * ux[diff]))
            g_diff_y = float(np.sum(wd * uy[diff]))

        # scale by thetas (can be negative)
        g_same_x *= self.theta_same
        g_same_y *= self.theta_same
        g_diff_x *= self.theta_diff
        g_diff_y *= self.theta_diff

        return g_same_x, g_same_y, g_diff_x, g_diff_y

    def _repulsion_short(self, uni: Universe, xs: np.ndarray, ys: np.ndarray, self_idx: int) -> tuple[float, float]:
        n = xs.size
        if n <= 1:
            return 0.0, 0.0

        dx, dy = torus_delta_vec(xs, ys, self.x, self.y, uni.size_x, uni.size_y)
        d2 = dx * dx + dy * dy
        d2[self_idx] = np.inf

        r2 = AGENT_REPULSION_RADIUS ** 2
        mask = d2 <= r2
        if not np.any(mask):
            return 0.0, 0.0

        d = np.sqrt(d2[mask])
        wgt = gaussian_repulsion(d, AGENT_REPULSION_RADIUS)
        inv = 1.0 / (d + EPS_DIR)
        ux = dx[mask] * inv
        uy = dy[mask] * inv
        # push away
        gx = -THETA_AGENT_REPULSION * float(np.sum(wgt * ux))
        gy = -THETA_AGENT_REPULSION * float(np.sum(wgt * uy))
        return gx, gy

    def propose_move(self, uni: Universe, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, self_idx: int) -> tuple[float, float]:
        # visible cues
        a_w_vis_x, a_w_vis_y, a_f_vis_x, a_f_vis_y = self._visible_vectors(uni)

        # urgency weights
        pi_w, pi_f = self.urgency_weights()

        # water seek if not visible
        a_w_eff_x, a_w_eff_y = self._update_water_seek(a_w_vis_x, a_w_vis_y, pi_w)

        # food unit
        a_f_u_x, a_f_u_y = unit_vec(a_f_vis_x, a_f_vis_y, EPS_DIR)

        # foraging vector
        g_for_x = pi_w * a_w_eff_x + pi_f * a_f_u_x
        g_for_y = pi_w * a_w_eff_y + pi_f * a_f_u_y

        # social vectors (ring-limited) + short repulsion
        g_same_x, g_same_y, g_diff_x, g_diff_y = self._social_vectors(uni, xs, ys, ts, self_idx)
        g_rep_x, g_rep_y = self._repulsion_short(uni, xs, ys, self_idx)

        # satiety gate: when high, reduce foraging pressure (pero no lo anula)
        sat = self.satiety_joint()
        g_raw_x = (1.0 - 0.7 * sat) * g_for_x + g_same_x + g_diff_x + g_rep_x
        g_raw_y = (1.0 - 0.7 * sat) * g_for_y + g_same_y + g_diff_y + g_rep_y

        # anti-paralysis exploration (solo si norma es baja)
        raw_norm = safe_norm(g_raw_x, g_raw_y, EPS_DIR)
        omega = self.omega0 * (G1_CANCEL_SCALE / (raw_norm + G1_CANCEL_SCALE))
        g_tot_x = g_raw_x + omega * self.xi_x
        g_tot_y = g_raw_y + omega * self.xi_y

        gux, guy = unit_vec(g_tot_x, g_tot_y, EPS_DIR)

        # heading inertia
        self.hx = (1.0 - self.alpha_h) * self.hx + self.alpha_h * gux
        self.hy = (1.0 - self.alpha_h) * self.hy + self.alpha_h * guy
        dir_x, dir_y = unit_vec(self.hx, self.hy, EPS_DIR)

        # propose displacement
        return self.step * dir_x, self.step * dir_y

    def try_eat(self, uni: Universe) -> bool:
        if not uni.food_x.size:
            return False
        d2 = torus_dist2(uni.food_x, uni.food_y, self.x, self.y, uni.size_x, uni.size_y)
        i = int(np.argmin(d2))
        if d2[i] <= FOOD_EAT_RADIUS ** 2:
            self.f = min(F_MAX, self.f + EAT_GAIN_F)
            uni.remove_food_index(i)
            return True
        return False

    def try_drink(self, uni: Universe) -> bool:
        if not uni.water_sources.size:
            return False
        wx = uni.water_sources[:, 0]
        wy = uni.water_sources[:, 1]
        d2 = torus_dist2(wx, wy, self.x, self.y, uni.size_x, uni.size_y)
        if float(np.min(d2)) <= WATER_DRINK_RADIUS ** 2:
            self.w = min(W_MAX, self.w + DRINK_GAIN_W)
            return True
        return False

    def metabolize(self):
        # baseline decays
        self.w -= BASE_W_DECAY
        self.f -= BASE_F_DECAY

        # cost per step (NOT distance)
        self.w -= MOVE_COST_PER_STEP
        self.f -= MOVE_COST_PER_STEP

        if self.w < LOW_W_THRESHOLD:
            self.f -= LOW_W_PENALTY_F

        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

        if self.w <= 0.0 or self.f <= 0.0:
            self.alive = False

# ----------------------------
# Population
# ----------------------------
class Population:
    def __init__(self, cells: list[Cell], uni: Universe):
        self.cells = list(cells)
        self.uni = uni

        self.step_count = 0

        self.births_total = 0
        self.deaths_total = 0
        self.births_by_type = {TYPE_A: 0, TYPE_B: 0}
        self.deaths_by_type = {TYPE_A: 0, TYPE_B: 0}

    def _alive_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.alive]

    def _snapshot(self, alive_cells: list[Cell]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array([c.x for c in alive_cells], dtype=float)
        ys = np.array([c.y for c in alive_cells], dtype=float)
        ts = np.array([c.type for c in alive_cells], dtype=int)
        return xs, ys, ts

    def step(self) -> dict:
        self.step_count += 1

        alive_cells = self._alive_cells()
        if not alive_cells:
            self.cells = []
            return {
                "step": self.step_count,
                "alive": 0,
                "alive_A": 0,
                "alive_B": 0,
                "births_step_A": 0,
                "births_step_B": 0,
                "deaths_step_A": 0,
                "deaths_step_B": 0,
                "births_total": self.births_total,
                "deaths_total": self.deaths_total,
            }

        # counts pre-step (para tasas)
        pre_A = sum(1 for c in alive_cells if c.type == TYPE_A)
        pre_B = sum(1 for c in alive_cells if c.type == TYPE_B)

        xs0, ys0, ts0 = self._snapshot(alive_cells)

        # (1) move proposals using snapshot
        N = len(alive_cells)
        order = np.arange(N)
        np.random.shuffle(order)

        # apply displacements
        for idx in order:
            c = alive_cells[idx]
            if not c.alive:
                continue
            dx, dy = c.propose_move(self.uni, xs0, ys0, ts0, idx)
            c.x, c.y = wrap_pos(c.x + dx, c.y + dy, self.uni.size_x, self.uni.size_y)

        # (2) hard collisions in batch (fast grid)
        xs1 = np.array([c.x for c in alive_cells], dtype=float)
        ys1 = np.array([c.y for c in alive_cells], dtype=float)
        xs1, ys1 = resolve_collisions_grid(
            xs1, ys1,
            self.uni.size_x, self.uni.size_y,
            r_col=COLLISION_DIST,
            cell_size=COLLISION_GRID_CELL,
            n_iters=COLLISION_ITERS
        )
        for i, c in enumerate(alive_cells):
            c.x = float(xs1[i])
            c.y = float(ys1[i])

        # (3) eat/drink
        for c in alive_cells:
            if not c.alive:
                continue
            c.try_eat(self.uni)
            c.try_drink(self.uni)

        # (4) reproduction (partner-based, anywhere)
        births_step_A = births_step_B = 0
        new_cells: list[Cell] = []
        reproduced = set()

        if ENABLE_REPRODUCTION:
            # snapshot after collisions for partner search
            xs2, ys2, ts2 = self._snapshot(alive_cells)

            order2 = np.arange(N)
            np.random.shuffle(order2)

            for i in order2:
                if i in reproduced:
                    continue
                c = alive_cells[i]
                if (not c.alive) or c.repro_cooldown > 0:
                    continue

                if c.w < REPRO_W_MIN or c.f < REPRO_F_MIN:
                    continue

                sat_i = c.satiety_joint()
                if sat_i <= 0.05:
                    continue

                dx, dy = torus_delta_vec(xs2, ys2, c.x, c.y, self.uni.size_x, self.uni.size_y)
                d2 = dx * dx + dy * dy
                d2[i] = np.inf

                re2 = INTERACTION_RADIUS ** 2
                mask = d2 <= re2
                if not np.any(mask):
                    continue

                cand = np.where(mask)[0]
                # elige pareja más cercana disponible
                cand = cand[np.argsort(d2[cand])]

                partner_idx = None
                for j in cand:
                    if j in reproduced:
                        continue
                    p = alive_cells[int(j)]
                    if (not p.alive) or p.repro_cooldown > 0:
                        continue
                    if p.w < REPRO_W_MIN or p.f < REPRO_F_MIN:
                        continue
                    sat_j = p.satiety_joint()
                    if sat_j <= 0.05:
                        continue
                    partner_idx = int(j)
                    break

                if partner_idx is None:
                    continue

                partner = alive_cells[partner_idx]
                sat_j = partner.satiety_joint()

                # prob
                p_rep = REPRO_BETA * sat_i * sat_j
                if c.rng.uniform(0.0, 1.0) >= p_rep:
                    continue

                # apply cooldown + costs
                c.repro_cooldown = REPRO_COOLDOWN_STEPS
                partner.repro_cooldown = REPRO_COOLDOWN_STEPS
                reproduced.add(i)
                reproduced.add(partner_idx)

                c.w = max(0.0, c.w - REPRO_COST_W)
                c.f = max(0.0, c.f - REPRO_COST_F)
                partner.w = max(0.0, partner.w - REPRO_COST_W)
                partner.f = max(0.0, partner.f - REPRO_COST_F)

                # child: "paquete completo" 50% de un padre
                donor = c if c.rng.uniform(0.0, 1.0) < 0.5 else partner
                child_type = donor.type

                # mutate only theta_same/theta_diff
                ths = donor.theta_same + c.rng.normal(0.0, THETA_MUT_SIGMA)
                thd = donor.theta_diff + c.rng.normal(0.0, THETA_MUT_SIGMA)
                ths = float(np.clip(ths, THETA_MIN, THETA_MAX))
                thd = float(np.clip(thd, THETA_MIN, THETA_MAX))

                ox = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
                oy = c.rng.uniform(-CHILD_SPAWN_JITTER, CHILD_SPAWN_JITTER)
                cx, cy = wrap_pos(c.x + ox, c.y + oy, self.uni.size_x, self.uni.size_y)

                child = Cell(cx, cy, child_type, w=min(W_MAX, 0.5*(c.w+partner.w)), f=min(F_MAX, 0.5*(c.f+partner.f)), rng=c.rng)
                child.theta_same = ths
                child.theta_diff = thd

                new_cells.append(child)
                if child_type == TYPE_A:
                    births_step_A += 1
                else:
                    births_step_B += 1

        # (5) metabolize + deaths
        deaths_step_A = deaths_step_B = 0
        for c in alive_cells:
            if not c.alive:
                continue
            c.metabolize()
            if not c.alive:
                if c.type == TYPE_A:
                    deaths_step_A += 1
                else:
                    deaths_step_B += 1

        # finalize
        self.cells = [c for c in self.cells if c.alive] + new_cells

        self.births_total += (births_step_A + births_step_B)
        self.deaths_total += (deaths_step_A + deaths_step_B)
        self.births_by_type[TYPE_A] += births_step_A
        self.births_by_type[TYPE_B] += births_step_B
        self.deaths_by_type[TYPE_A] += deaths_step_A
        self.deaths_by_type[TYPE_B] += deaths_step_B

        alive_now = self._alive_cells()
        alive_A = sum(1 for c in alive_now if c.type == TYPE_A)
        alive_B = sum(1 for c in alive_now if c.type == TYPE_B)

        return {
            "step": self.step_count,
            "alive": len(alive_now),
            "alive_A": alive_A,
            "alive_B": alive_B,
            "pre_A": pre_A,
            "pre_B": pre_B,
            "births_step_A": births_step_A,
            "births_step_B": births_step_B,
            "deaths_step_A": deaths_step_A,
            "deaths_step_B": deaths_step_B,
            "births_total": self.births_total,
            "deaths_total": self.deaths_total,
        }

    def arrays(self):
        alive = self._alive_cells()
        if not alive:
            return (np.empty(0),)*7
        x = np.array([c.x for c in alive], dtype=float)
        y = np.array([c.y for c in alive], dtype=float)
        t = np.array([c.type for c in alive], dtype=int)
        w = np.array([c.w for c in alive], dtype=float)
        f = np.array([c.f for c in alive], dtype=float)
        E = np.array([c.E for c in alive], dtype=float)
        ths = np.array([c.theta_same for c in alive], dtype=float)
        thd = np.array([c.theta_diff for c in alive], dtype=float)
        return x, y, t, w, f, E, ths, thd

# ----------------------------
# Simulation + Viz
# ----------------------------
class Simulation:
    def __init__(self, uni: Universe, pop: Population):
        self.uni = uni
        self.pop = pop
        self._ani = None

    def animate(self, steps: int = 4000, interval_ms: int = 16):
        plt.style.use("dark_background")

        fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="black")
        gs = GridSpec(1, 2, width_ratios=[3.3, 1.0], wspace=0.05)

        ax = fig.add_subplot(gs[0, 0])
        ax_stats = fig.add_subplot(gs[0, 1])

        # world axis
        ax.set_facecolor("black")
        ax.set_xlim(0, self.uni.size_x)
        ax.set_ylim(0, self.uni.size_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_alpha(0.25)

        # stats axis
        ax_stats.set_facecolor("black")
        ax_stats.set_xticks([])
        ax_stats.set_yticks([])
        for sp in ax_stats.spines.values():
            sp.set_alpha(0.25)

        # background stars
        stars = np.random.uniform([0, 0], [self.uni.size_x, self.uni.size_y], size=(350, 2))
        ax.scatter(stars[:, 0], stars[:, 1], s=2, alpha=0.15)

        # water sources
        for wx, wy in self.uni.water_sources:
            ax.add_patch(plt.Circle((wx, wy), WATER_SOURCES_RADIUS, alpha=0.18))
            ax.add_patch(plt.Circle((wx, wy), WATER_SOURCES_RADIUS * 0.55, alpha=0.25))
            ax.add_patch(plt.Circle((wx, wy), WATER_DRINK_RADIUS, fill=False, linestyle=":", linewidth=0.9, alpha=0.35))

        # food
        food_sc = ax.scatter([], [], s=10, marker=".", alpha=0.9)

        # two scatters for types (distinct colors)
        a_sc = ax.scatter([], [], s=28, marker="o", alpha=0.95, c="#4CC9F0")  # A
        b_sc = ax.scatter([], [], s=28, marker="o", alpha=0.95, c="#F4A261")  # B

        # stats text (single artist)
        stats_text = ax_stats.text(
            0.03, 0.98, "",
            transform=ax_stats.transAxes,
            ha="left", va="top",
            fontsize=10,
            family="monospace",
            alpha=0.95
        )

        def _format_stats(frame: int, stats: dict) -> str:
            x, y, t, w, f, E, ths, thd = self.pop.arrays()
            alive = stats["alive"]
            alive_A = stats["alive_A"]
            alive_B = stats["alive_B"]
            pre_A = stats.get("pre_A", max(1, alive_A))
            pre_B = stats.get("pre_B", max(1, alive_B))

            # rates per step (approx)
            mort_A = stats["deaths_step_A"] / max(1, pre_A)
            mort_B = stats["deaths_step_B"] / max(1, pre_B)
            birth_A = stats["births_step_A"] / max(1, pre_A)
            birth_B = stats["births_step_B"] / max(1, pre_B)

            prop_A = alive_A / max(1, alive)
            prop_B = alive_B / max(1, alive)

            if alive > 0:
                avg_w = w.mean()
                avg_f = f.mean()
                avg_E = E.mean()
            else:
                avg_w = avg_f = avg_E = 0.0

            # thetas per type
            if alive > 0:
                maskA = (t == TYPE_A)
                maskB = (t == TYPE_B)

                def m_or0(arr, m):
                    return float(arr[m].mean()) if np.any(m) else 0.0

                ths_A = m_or0(ths, maskA)
                thd_A = m_or0(thd, maskA)
                ths_B = m_or0(ths, maskB)
                thd_B = m_or0(thd, maskB)
                ths_all = float(ths.mean())
                thd_all = float(thd.mean())
            else:
                ths_A = thd_A = ths_B = thd_B = ths_all = thd_all = 0.0

            lines = [
                f"Step: {frame}",
                "",
                f"Alive: {alive:4d}",
                f"  A: {alive_A:4d}   ({prop_A:5.1%})",
                f"  B: {alive_B:4d}   ({prop_B:5.1%})",
                "",
                f"Births (step):  A {stats['births_step_A']:3d}  |  B {stats['births_step_B']:3d}",
                f"Deaths (step):  A {stats['deaths_step_A']:3d}  |  B {stats['deaths_step_B']:3d}",
                f"Birth rate:     A {birth_A:6.2%} |  B {birth_B:6.2%}",
                f"Mortality:      A {mort_A:6.2%} |  B {mort_B:6.2%}",
                "",
                f"Births total: {stats['births_total']:5d}",
                f"Deaths total: {stats['deaths_total']:5d}",
                "",
                f"Avg stocks:",
                f"  w: {avg_w:6.2f}",
                f"  f: {avg_f:6.2f}",
                f"  E(w,f): {avg_E:6.2f}",
                "",
                f"Theta means:",
                f"  A: same {ths_A:+.4f}   diff {thd_A:+.4f}",
                f"  B: same {ths_B:+.4f}   diff {thd_B:+.4f}",
                f" all: same {ths_all:+.4f}   diff {thd_all:+.4f}",
            ]
            return "\n".join(lines)

        def init():
            if self.uni.food_x.size:
                food_sc.set_offsets(np.column_stack([self.uni.food_x, self.uni.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            x, y, t, *_ = self.pop.arrays()
            if x.size:
                a_xy = np.column_stack([x[t == TYPE_A], y[t == TYPE_A]])
                b_xy = np.column_stack([x[t == TYPE_B], y[t == TYPE_B]])
                a_sc.set_offsets(a_xy if a_xy.size else np.empty((0, 2)))
                b_sc.set_offsets(b_xy if b_xy.size else np.empty((0, 2)))
            else:
                a_sc.set_offsets(np.empty((0, 2)))
                b_sc.set_offsets(np.empty((0, 2)))

            stats = {
                "alive": len(self.pop._alive_cells()),
                "alive_A": sum(1 for c in self.pop._alive_cells() if c.type == TYPE_A),
                "alive_B": sum(1 for c in self.pop._alive_cells() if c.type == TYPE_B),
                "pre_A": 1, "pre_B": 1,
                "births_step_A": 0, "births_step_B": 0,
                "deaths_step_A": 0, "deaths_step_B": 0,
                "births_total": self.pop.births_total,
                "deaths_total": self.pop.deaths_total,
            }
            stats_text.set_text(_format_stats(0, stats))
            return food_sc, a_sc, b_sc, stats_text

        def update(frame: int):
            # environment
            if self.uni.food_x.size < MAX_FOOD:
                self.uni.spawn_food(FOOD_SPAWN_PER_STEP)

            stats = self.pop.step()

            # food
            if self.uni.food_x.size:
                food_sc.set_offsets(np.column_stack([self.uni.food_x, self.uni.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            # agents by type
            x, y, t, *_ = self.pop.arrays()
            if x.size:
                a_xy = np.column_stack([x[t == TYPE_A], y[t == TYPE_A]])
                b_xy = np.column_stack([x[t == TYPE_B], y[t == TYPE_B]])
                a_sc.set_offsets(a_xy if a_xy.size else np.empty((0, 2)))
                b_sc.set_offsets(b_xy if b_xy.size else np.empty((0, 2)))
            else:
                a_sc.set_offsets(np.empty((0, 2)))
                b_sc.set_offsets(np.empty((0, 2)))

            stats_text.set_text(_format_stats(frame, stats))
            return food_sc, a_sc, b_sc, stats_text

        self._ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval_ms)
        plt.show()

# ----------------------------
# Runner
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    uni = Universe(rng=rng)
    uni.gen_water_sources()
    uni.spawn_food(MAX_FOOD // 2)

    cells: list[Cell] = []
    # inicial: 100 A, 100 B (ajusta)
    for _ in range(100):
        x = rng.uniform(0, uni.size_x)
        y = rng.uniform(0, uni.size_y)
        cells.append(Cell(x, y, TYPE_A, w=18.0, f=18.0, rng=rng))
    for _ in range(100):
        x = rng.uniform(0, uni.size_x)
        y = rng.uniform(0, uni.size_y)
        cells.append(Cell(x, y, TYPE_B, w=18.0, f=18.0, rng=rng))

    pop = Population(cells, uni)
    sim = Simulation(uni, pop)
    sim.animate(steps=4000, interval_ms=16)
