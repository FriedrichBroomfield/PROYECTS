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
    REPRO_GOAL=170.0,         # si alcanza durante el día -> vuelve y espera
    SURVIVE_GOAL=60.0,        # si queda poco día y ya estás seguro -> vuelve

    # Energetic satiety (simplificación clave)
    SATIETY_E=220.0,          # >= E_INIT_MAX para que nadie "nazca saciado"
    SATIETY_EPS=1e-3,         # tolerancia pequeña

    # Hawk-Dove
    INJURY_COST_D=21.0,       # D (H-H pierde uno)
    TIME_COST_T=10.0,         # T (D-D ambos)

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


# ============================================================
# Environment
# ============================================================
class Environment:
    def __init__(self):
        self.r_max = float(P["R_MAX"])
        self.food_x = cp.array([], dtype=cp.float32)
        self.food_y = cp.array([], dtype=cp.float32)

    @staticmethod
    def cart_to_polar(x, y):
        r = cp.sqrt(x * x + y * y)
        th = cp.arctan2(y, x) % (2 * cp.pi)
        return r, th

    def set_food_for_day(self):
        self.food_x = cp.array([], dtype=cp.float32)
        self.food_y = cp.array([], dtype=cp.float32)
        self.spawn_food(P["FOOD_PER_DAY"])

    def clear_food(self):
        self.food_x = cp.array([], dtype=cp.float32)
        self.food_y = cp.array([], dtype=cp.float32)

    def spawn_food(self, k: int):
        if k <= 0:
            return
        r_min = max(P["FOOD_MARGIN"], P["FOOD_SPAWN_R_MIN"])
        r_max = self.r_max - P["FOOD_MARGIN"]
        if r_min >= r_max:
            return

        # uniform in area annulus
        u = cp.random.random(k, dtype=cp.float32)
        r = cp.sqrt(u * (r_max * r_max - r_min * r_min) + r_min * r_min)
        th = cp.random.random(k, dtype=cp.float32) * (2 * cp.pi)
        x = r * cp.cos(th)
        y = r * cp.sin(th)
        self.food_x = cp.concatenate([self.food_x, x.astype(cp.float32)])
        self.food_y = cp.concatenate([self.food_y, y.astype(cp.float32)])

    def consume_food_indices(self, eaten_idx_cpu):
        if not eaten_idx_cpu:
            return
        m = int(self.food_x.size)
        mask = np.ones(m, dtype=bool)
        mask[np.array(eaten_idx_cpu, dtype=int)] = False
        keep = cp.asarray(mask)
        self.food_x = self.food_x[keep]
        self.food_y = self.food_y[keep]

    def nearest_food(self, cell_x, cell_y):
        m = int(self.food_x.size)
        if m == 0:
            return None, None
        dx = self.food_x[None, :] - cell_x[:, None]
        dy = self.food_y[None, :] - cell_y[:, None]
        d2 = dx * dx + dy * dy
        idx = cp.argmin(d2, axis=1).astype(cp.int32)
        min_d = cp.sqrt(d2[cp.arange(d2.shape[0]), idx])
        return idx, min_d


# ============================================================
# Cell contests: Hawk vs Dove (CPU tournament for clarity)
# ============================================================
class Cell:
    HAWK = 1
    DOVE = 0

    @staticmethod
    def pair_contest(a_idx, b_idx, types_cpu):
        a_h = types_cpu[a_idx] == Cell.HAWK
        b_h = types_cpu[b_idx] == Cell.HAWK

        injury = {}
        time = {}

        if a_h and b_h:
            if random.random() < 0.5:
                injury[b_idx] = injury.get(b_idx, 0.0) + P["INJURY_COST_D"]
                return a_idx, injury, time
            else:
                injury[a_idx] = injury.get(a_idx, 0.0) + P["INJURY_COST_D"]
                return b_idx, injury, time

        if a_h and (not b_h):
            return a_idx, injury, time
        if (not a_h) and b_h:
            return b_idx, injury, time

        time[a_idx] = time.get(a_idx, 0.0) + P["TIME_COST_T"]
        time[b_idx] = time.get(b_idx, 0.0) + P["TIME_COST_T"]
        return (a_idx if random.random() < 0.5 else b_idx), injury, time

    @staticmethod
    def tournament(contenders, types_cpu):
        contenders = contenders[:]
        random.shuffle(contenders)
        injury = {}
        time = {}
        while len(contenders) > 1:
            a = contenders.pop()
            b = contenders.pop()
            w, inj, tim = Cell.pair_contest(a, b, types_cpu)
            for k, v in inj.items():
                injury[k] = injury.get(k, 0.0) + v
            for k, v in tim.items():
                time[k] = time.get(k, 0.0) + v
            contenders.append(w)
        return contenders[0], injury, time


# ============================================================
# Population
# ============================================================
class Population:
    def __init__(self, env: Environment):
        self.env = env
        self.x = cp.array([], dtype=cp.float32)
        self.y = cp.array([], dtype=cp.float32)
        self.energy = cp.array([], dtype=cp.float32)
        self.type = cp.array([], dtype=cp.int8)

        # Daily state
        self.done_for_day = cp.array([], dtype=cp.bool_)    # volvió/espera
        self.ready_to_repro = cp.array([], dtype=cp.bool_)  # alcanzó REPRO_GOAL hoy
        self.ate_today = cp.array([], dtype=cp.bool_)       # comió al menos 1 vez

        # Ledgers (causas muerte)
        self.cost_day = cp.array([], dtype=cp.float32)
        self.cost_night = cp.array([], dtype=cp.float32)
        self.cost_fight_injury = cp.array([], dtype=cp.float32)
        self.cost_fight_time = cp.array([], dtype=cp.float32)

    def size(self):
        return int(self.energy.size)

    def init_random(self):
        n0 = int(P["N0"])
        th = cp.random.random(n0, dtype=cp.float32) * (2 * cp.pi)
        r = cp.full(n0, cp.float32(self.env.r_max), dtype=cp.float32)
        self.x = r * cp.cos(th)
        self.y = r * cp.sin(th)

        self.energy = cp.asarray(
            np.random.uniform(P["E_INIT_MIN"], P["E_INIT_MAX"], n0),
            dtype=cp.float32
        )
        self.type = cp.asarray((np.random.random(n0) < P["HACK_RATIO0"]).astype(np.int8))

        self.done_for_day = cp.zeros(n0, dtype=cp.bool_)
        self.ready_to_repro = cp.zeros(n0, dtype=cp.bool_)
        self.ate_today = cp.zeros(n0, dtype=cp.bool_)

        z = cp.zeros(n0, dtype=cp.float32)
        self.cost_day = z.copy()
        self.cost_night = z.copy()
        self.cost_fight_injury = z.copy()
        self.cost_fight_time = z.copy()

    def reset_day_flags(self):
        self.done_for_day[:] = False
        self.ready_to_repro[:] = False
        self.ate_today[:] = False

    def metabolic(self, day: bool):
        c = cp.float32(P["METABOLIC_DAY"] if day else P["METABOLIC_NIGHT"])
        self.energy -= c
        if day:
            self.cost_day += c
        else:
            self.cost_night += c

    def move_cost(self, step, day: bool, mask=None):
        c = cp.float32(P["MOVE_COST_PER_DIST"]) * step
        if mask is None:
            self.energy -= c
            if day:
                self.cost_day += c
            else:
                self.cost_night += c
        else:
            self.energy -= cp.where(mask, c, 0.0)
            if day:
                self.cost_day += cp.where(mask, c, 0.0)
            else:
                self.cost_night += cp.where(mask, c, 0.0)

    def reproduce_at_dawn(self):
        """Reproduce only at nest, if energy >= REPRO_THRESHOLD. Child appended."""
        if self.size() == 0:
            return 0

        r = cp.sqrt(self.x * self.x + self.y * self.y)
        at_nest = r >= cp.float32(self.env.r_max - P["NEST_THICKNESS"])

        parents = cp.where(at_nest & (self.energy >= cp.float32(P["REPRO_THRESHOLD"])))[0]
        if int(parents.size) == 0:
            return 0

        room = int(P["POP_CAP"]) - self.size()
        if room <= 0:
            return 0

        parents = parents[: min(room, int(parents.size))]
        k = int(parents.size)

        split = cp.float32(P["REPRO_SPLIT"])
        child_energy = self.energy[parents] * split
        self.energy[parents] = self.energy[parents] * split

        th_parent = cp.arctan2(self.y[parents], self.x[parents]) % (2 * cp.pi)
        th_child = (th_parent + (cp.random.random(k, dtype=cp.float32) - 0.5) * 0.28) % (2 * cp.pi)

        r_child = cp.full(k, cp.float32(self.env.r_max), dtype=cp.float32)
        x_child = r_child * cp.cos(th_child)
        y_child = r_child * cp.sin(th_child)

        self.x = cp.concatenate([self.x, x_child.astype(cp.float32)])
        self.y = cp.concatenate([self.y, y_child.astype(cp.float32)])
        self.energy = cp.concatenate([self.energy, child_energy.astype(cp.float32)])
        self.type = cp.concatenate([self.type, self.type[parents].astype(cp.int8)])

        self.done_for_day = cp.concatenate([self.done_for_day, cp.zeros(k, dtype=cp.bool_)])
        self.ready_to_repro = cp.concatenate([self.ready_to_repro, cp.zeros(k, dtype=cp.bool_)])
        self.ate_today = cp.concatenate([self.ate_today, cp.zeros(k, dtype=cp.bool_)])

        self.cost_day = cp.concatenate([self.cost_day, cp.zeros(k, dtype=cp.float32)])
        self.cost_night = cp.concatenate([self.cost_night, cp.zeros(k, dtype=cp.float32)])
        self.cost_fight_injury = cp.concatenate([self.cost_fight_injury, cp.zeros(k, dtype=cp.float32)])
        self.cost_fight_time = cp.concatenate([self.cost_fight_time, cp.zeros(k, dtype=cp.float32)])

        return k

    def kill_dead_and_classify(self, counters):
        dead = self.energy <= 0
        dead_idx = cp.where(dead)[0]
        dead_n = int(dead_idx.size)

        if dead_n > 0:
            costs = cp.stack([
                self.cost_day[dead_idx],
                self.cost_night[dead_idx],
                self.cost_fight_injury[dead_idx],
                self.cost_fight_time[dead_idx],
            ], axis=1)
            cause = cp.argmax(costs, axis=1)
            names = ["day_cost", "night_cost", "fight_injury", "fight_time"]
            for c in to_cpu(cause).tolist():
                counters[names[int(c)]] = counters.get(names[int(c)], 0) + 1

        alive = ~dead
        self.x = self.x[alive]
        self.y = self.y[alive]
        self.energy = self.energy[alive]
        self.type = self.type[alive]
        self.done_for_day = self.done_for_day[alive]
        self.ready_to_repro = self.ready_to_repro[alive]
        self.ate_today = self.ate_today[alive]
        self.cost_day = self.cost_day[alive]
        self.cost_night = self.cost_night[alive]
        self.cost_fight_injury = self.cost_fight_injury[alive]
        self.cost_fight_time = self.cost_fight_time[alive]
        return dead_n


# ============================================================
# Simulation (global day/night) + FIX de transiciones visibles
# ============================================================
class Simulation:
    def __init__(self):
        self.env = Environment()
        self.pop = Population(self.env)
        self.pop.init_random()

        self.phase = "night"
        self.phase_step = 0
        self.day = 0

        self.stats = dict(
            births_today=0,
            deaths_today=0,
            eaten_today=0,
            death_causes={}
        )

        self.just_started_phase = True
        self.start_day()  # arrancamos en día 1

    def start_day(self):
        # Dawn: reproduce, reset flags, new apples
        self.stats["births_today"] = self.pop.reproduce_at_dawn()
        self.pop.reset_day_flags()
        self.env.set_food_for_day()

        self.stats["eaten_today"] = 0
        self.stats["deaths_today"] = 0

        # IMPORTANTE: NO "push inside" aquí.
        # Así el primer frame del día muestra a todos en el nido (outer ring),
        # y los movimientos se ven solo cuando haces step_day.

        self.phase = "day"
        self.phase_step = 0
        self.day += 1
        self.just_started_phase = True

    def start_night(self):
        # Dusk: leftover food disappears
        self.env.clear_food()
        self.phase = "night"
        self.phase_step = 0
        self.just_started_phase = True

    def step(self):
        """
        Avanza 1 tick.
        Retorna "dawn" o "dusk" cuando hay transición (para cortar batch en animate).
        """
        if self.pop.size() == 0:
            return None

        if self.phase == "day":
            self.step_day()
            self.phase_step += 1
            if self.phase_step >= P["DAY_STEPS"]:
                self.start_night()
                return "dusk"
        else:
            self.step_night()
            self.phase_step += 1
            if self.phase_step >= P["NIGHT_STEPS"]:
                self.start_day()
                return "dawn"

        return None

    def step_night(self):
        # everyone returns together
        self.pop.metabolic(day=False)

        dx, dy, step = radial_out_step(self.pop.x, self.pop.y, self.env.r_max, P["SPEED"])
        self.pop.x += dx
        self.pop.y += dy
        self.pop.x, self.pop.y = clip_to_circle(self.pop.x, self.pop.y, self.env.r_max)
        self.pop.move_cost(step, day=False)

        d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
        self.stats["deaths_today"] += d

    def step_day(self):
        self.pop.metabolic(day=True)

        # energetic satiety: if already >= SATIETY_E -> stop searching today
        self.pop.done_for_day |= (self.pop.energy >= cp.float32(P["SATIETY_E"] - P["SATIETY_EPS"]))

        # returners move outward to nest (so they're consistent with "wait at nest")
        returning = self.pop.done_for_day
        if bool(to_cpu(cp.any(returning))):
            dxr, dyr, stepr = radial_out_step(self.pop.x, self.pop.y, self.env.r_max, P["SPEED"])
            self.pop.x = cp.where(returning, self.pop.x + dxr, self.pop.x)
            self.pop.y = cp.where(returning, self.pop.y + dyr, self.pop.y)
            self.pop.x, self.pop.y = clip_to_circle(self.pop.x, self.pop.y, self.env.r_max)
            self.pop.move_cost(stepr, day=True, mask=returning)

        seekers = ~self.pop.done_for_day
        if not bool(to_cpu(cp.any(seekers))):
            d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
            self.stats["deaths_today"] += d
            return

        idx_food, dist_food = self.env.nearest_food(self.pop.x, self.pop.y)
        if idx_food is None:
            dx, dy, step = brownian_step(self.pop.size(), P["BROWNIAN_SIGMA"], P["SPEED"])
            self.pop.x = cp.where(seekers, self.pop.x + dx, self.pop.x)
            self.pop.y = cp.where(seekers, self.pop.y + dy, self.pop.y)
            self.pop.x, self.pop.y = clip_to_circle(self.pop.x, self.pop.y, self.env.r_max)
            self.pop.move_cost(step, day=True, mask=seekers)

            d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
            self.stats["deaths_today"] += d
            return

        detect = seekers & (dist_food < cp.float32(P["SENSORY_RANGE"] * P["SENSORY_FACTOR"]))
        fx = self.env.food_x[idx_food]
        fy = self.env.food_y[idx_food]

        dx_t, dy_t, step_t = towards_step(self.pop.x, self.pop.y, fx, fy, P["SPEED"], noise_angle=0.10)
        dx_b, dy_b, step_b = brownian_step(self.pop.size(), P["BROWNIAN_SIGMA"], P["SPEED"])

        dx = cp.where(detect, dx_t, cp.where(seekers, dx_b, 0.0))
        dy = cp.where(detect, dy_t, cp.where(seekers, dy_b, 0.0))
        step = cp.where(detect, step_t, cp.where(seekers, step_b, 0.0))

        self.pop.x += dx
        self.pop.y += dy
        self.pop.x, self.pop.y = clip_to_circle(self.pop.x, self.pop.y, self.env.r_max)
        self.pop.move_cost(step, day=True, mask=seekers)

        # grab candidates: only hungry compete for food
        hungry = self.pop.energy < cp.float32(P["SATIETY_E"] - P["SATIETY_EPS"])
        fx2 = self.env.food_x[idx_food]
        fy2 = self.env.food_y[idx_food]
        ddx = fx2 - self.pop.x
        ddy = fy2 - self.pop.y
        dist2 = cp.sqrt(ddx * ddx + ddy * ddy)
        can_grab = detect & hungry & (dist2 < cp.float32(P["GRAB_RADIUS"]))

        cand_cells = to_cpu(cp.where(can_grab)[0])
        cand_food = to_cpu(idx_food[can_grab])

        buckets = {}
        for ci, fj in zip(cand_cells.tolist(), cand_food.tolist()):
            buckets.setdefault(int(fj), []).append(int(ci))

        eaten_idx = []
        if buckets:
            types_cpu = to_cpu(self.pop.type).astype(int)
            injury_delta = {}
            time_delta = {}
            winners = []

            for food_j, group in buckets.items():
                champ, inj, tim = Cell.tournament(group, types_cpu)
                winners.append(champ)
                eaten_idx.append(food_j)
                for k, v in inj.items():
                    injury_delta[k] = injury_delta.get(k, 0.0) + v
                for k, v in tim.items():
                    time_delta[k] = time_delta.get(k, 0.0) + v

            if injury_delta:
                idxs = cp.asarray(list(injury_delta.keys()), dtype=cp.int32)
                dvs = cp.asarray(list(injury_delta.values()), dtype=cp.float32)
                self.pop.energy[idxs] -= dvs
                self.pop.cost_fight_injury[idxs] += dvs

            if time_delta:
                idxs = cp.asarray(list(time_delta.keys()), dtype=cp.int32)
                dvs = cp.asarray(list(time_delta.values()), dtype=cp.float32)
                self.pop.energy[idxs] -= dvs
                self.pop.cost_fight_time[idxs] += dvs

            # winners eat: absorb only up to SATIETY_E (no overflow)
            if winners:
                win_idx = cp.asarray(winners, dtype=cp.int32)
                need = cp.float32(P["SATIETY_E"]) - self.pop.energy[win_idx]
                gain = cp.minimum(cp.float32(P["FOOD_VALUE_V"]), cp.maximum(need, 0.0))
                self.pop.energy[win_idx] += gain
                self.pop.ate_today[win_idx] = True
                self.stats["eaten_today"] += len(winners)

                # daily priority 1: if reached repro goal, stop and return
                reached = self.pop.energy[win_idx] >= cp.float32(P["REPRO_GOAL"])
                reached_idx = win_idx[reached]
                if int(reached_idx.size) > 0:
                    self.pop.ready_to_repro[reached_idx] = True
                    self.pop.done_for_day[reached_idx] = True

            self.env.consume_food_indices(eaten_idx)

        # daily priority 2: near end of day, if safe, stop and return
        remaining = (P["DAY_STEPS"] - (self.phase_step + 1))
        if remaining <= int(0.25 * P["DAY_STEPS"]):
            safe = (self.pop.energy >= cp.float32(P["SURVIVE_GOAL"]))
            can_stop = (~self.pop.done_for_day) & (~self.pop.ready_to_repro) & safe
            self.pop.done_for_day |= can_stop

        d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
        self.stats["deaths_today"] += d


# ============================================================
# Plot / UI
# ============================================================
random.seed(2)
np.random.seed(2)

sim = Simulation()
env = sim.env
pop = sim.pop

fig = plt.figure(figsize=(10.2, 7.2), facecolor="black")

ax_stats = fig.add_axes([0.02, 0.10, 0.30, 0.84], facecolor="black")
ax_stats.set_axis_off()
stats_txt = ax_stats.text(
    0.02, 0.98, "", va="top", ha="left",
    color="white", fontsize=10, family="monospace"
)

ax_slider = fig.add_axes([0.05, 0.04, 0.25, 0.03], facecolor="black")
speed_slider = Slider(ax_slider, "steps/frame", valmin=1, valmax=140, valinit=P["STEPS_PER_FRAME_INIT"], valstep=1)
speed_slider.label.set_color("white")
speed_slider.valtext.set_color("white")

ax = fig.add_axes([0.35, 0.06, 0.63, 0.88], projection="polar", facecolor="black")
ax.set_ylim(0, env.r_max)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["polar"].set_edgecolor("white")
ax.spines["polar"].set_linewidth(1.6)
ax.set_title("Hack vs Dove (essential + correct phase starts)", color="white", pad=18)

ring_th = np.linspace(0, TWOPI, 800)
ax.plot(ring_th, np.full_like(ring_th, env.r_max), color="white", linewidth=2.0, alpha=0.95)

food_sc = ax.scatter([], [], s=28, c="green", alpha=0.65)
cell_sc = ax.scatter([], [], s=20, c="dodgerblue", alpha=0.85)


def format_death_causes(dc: dict):
    total = sum(dc.values()) if dc else 0
    if total == 0:
        return "death causes: (no deaths yet)"
    order = sorted(dc.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["death causes (share):"]
    for k, v in order[:5]:
        lines.append(f"  {k:12s}: {v:5d} ({100.0*v/total:5.1f}%)")
    return "\n".join(lines)


def animate(_):
    # ========================================================
    # FIX CRÍTICO: mostrar inicio de fase SIN "movimientos invisibles"
    # - Si acaba de empezar día o noche: NO avanzamos steps.
    # - Si no: avanzamos un batch, pero CORTAMOS al cruzar una transición.
    # ========================================================
    if sim.just_started_phase:
        sim.just_started_phase = False
    else:
        steps = int(speed_slider.val)
        for _ in range(steps):
            event = sim.step()  # <<< UNA sola llamada por iteración
            if event is not None:
                # Cortamos para que el próximo frame muestre phase_step=0 limpio
                break
            if pop.size() == 0:
                break

    # food
    if int(env.food_x.size) > 0:
        fr, fth = env.cart_to_polar(env.food_x, env.food_y)
        food_sc.set_offsets(np.c_[to_cpu(fth), to_cpu(fr)])
        food_sc.set_sizes(np.full(int(env.food_x.size), 28.0))
    else:
        food_sc.set_offsets(np.empty((0, 2)))

    # cells
    if pop.size() == 0:
        cell_sc.set_offsets(np.empty((0, 2)))
        stats_txt.set_text("Extinción total.\n")
        return food_sc, cell_sc, stats_txt

    r, th = env.cart_to_polar(pop.x, pop.y)
    e = to_cpu(pop.energy)
    t = to_cpu(pop.type)
    done = to_cpu(pop.done_for_day)
    ready = to_cpu(pop.ready_to_repro)
    ate = to_cpu(pop.ate_today)

    colors = np.array(["red" if x == 1 else "dodgerblue" for x in t])
    sizes = np.clip(e, 10, 180)

    cell_sc.set_offsets(np.c_[to_cpu(th), to_cpu(r)])
    cell_sc.set_sizes(sizes)
    cell_sc.set_color(colors)

    total = pop.size()
    hawks = int((t == 1).sum())
    doves = total - hawks
    ph = 100.0 * hawks / total
    pd = 100.0 * doves / total

    pct_done = 100.0 * float(done.mean())
    pct_ready = 100.0 * float(ready.mean())
    pct_ate = 100.0 * float(ate.mean())
    pct_sated = 100.0 * float((e >= (P["SATIETY_E"] - P["SATIETY_EPS"])).mean())

    stats_txt.set_text(
        f"phase: {sim.phase}  step: {sim.phase_step}/{P['DAY_STEPS'] if sim.phase=='day' else P['NIGHT_STEPS']}\n"
        f"day: {sim.day}\n"
        f"pop: {total:4d}   births_today: {sim.stats['births_today']:4d}   deaths_today: {sim.stats['deaths_today']:4d}\n"
        f"Hawk: {hawks:4d} ({ph:5.1f}%)   Dove: {doves:4d} ({pd:5.1f}%)\n"
        f"E(mean/med): {e.mean():7.2f} / {np.median(e):7.2f}\n"
        f"food active: {int(env.food_x.size):4d}   eaten_today: {sim.stats['eaten_today']:4d}\n"
        f"\n"
        f"per-cell % (today)\n"
        f"  ate >=1      : {pct_ate:6.1f}%\n"
        f"  sated (E>=S) : {pct_sated:6.1f}%  (S={P['SATIETY_E']})\n"
        f"  done_for_day : {pct_done:6.1f}%\n"
        f"  ready_repro  : {pct_ready:6.1f}%  (goal={P['REPRO_GOAL']})\n"
        f"\n"
        f"V={P['FOOD_VALUE_V']}  D={P['INJURY_COST_D']}  T={P['TIME_COST_T']}\n"
        f"move_cost={P['MOVE_COST_PER_DIST']}/dist  metab(day/night)={P['METABOLIC_DAY']}/{P['METABOLIC_NIGHT']}\n"
        f"repro_thr={P['REPRO_THRESHOLD']}  food/day={P['FOOD_PER_DAY']}  r_min={P['FOOD_SPAWN_R_MIN']}\n"
        f"\n"
        f"{format_death_causes(sim.stats['death_causes'])}\n"
    )

    return food_sc, cell_sc, stats_txt


ani = FuncAnimation(fig, animate, frames=2500, interval=45, blit=False)
plt.show()

