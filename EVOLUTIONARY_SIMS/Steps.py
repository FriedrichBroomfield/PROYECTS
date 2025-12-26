import numpy as np
import cupy as cp
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


# ============================================================
# Config
# ============================================================
P = dict(
    # Mundo
    R_MAX=10.0,

    # Población
    POP_CAP=1000,
    N0=400,
    HACK_RATIO0=0.5,  # 0.0 = solo Doves, 1.0 = solo Hawks

    # Energía
    ENERGY_INIT_MIN=80.0,
    ENERGY_INIT_MAX=120.0,
    ENERGY_CAP=1000.0,

    # Movimiento
    SPEED=0.22,
    MOVE_COST_BASE=0.25,
    MOVE_COST_PER_STEP=0.20,
    RANDWALK_COST=0.35,

    # Nest / ciclo
    NEST_THICKNESS=0.25,
    RETURN_IF_LOW=28.0,

    # Reproducción (solo en nido)
    REPRO_THRESHOLD=120.0,   # (subido vs 70 para evitar "repro diluye energía" demasiado pronto)
    REPRO_FRACTION=0.50,     # energía que se transfiere al hijo (padre queda con 1-frac)

    # Comida unitaria
    FOOD_MAX=1000,
    FOOD_SPAWN_RATE=200.0,     # Poisson por step
    FOOD_UNIT_ENERGY=26.0,     # V (valor del recurso)
    FOOD_MARGIN=0.6,
    GRAB_RADIUS=0.42,

    # Evitar acumulación en el centro
    FOOD_SPAWN_R_MIN=2.0,      # radio mínimo donde spawnea comida (sube para vaciar el centro)
    FOOD_TTL=400,              # pasos antes de que una comida no consumida desaparezca

    # Sensado
    SENSORY_RANGE=2.2,
    SENSORY_FACTOR=5.0,

    # Hawk–Dove (pagos del diagrama)
    INJURY_COST_D=50.0,  # D
    TIME_COST_T=5.0,     # T
)

# Conveniencia
TWOPI = 2 * np.pi


def to_cpu(a):
    """CuPy -> NumPy para graficar."""
    # Manejar tuplas/listas (p. ej. cart_to_polar devuelve (r, th))
    if isinstance(a, (tuple, list)):
        out = []
        for x in a:
            if hasattr(x, "get"):
                out.append(x.get())
            else:
                try:
                    out.append(cp.asnumpy(x))
                except Exception:
                    out.append(x)
        return tuple(out) if isinstance(a, tuple) else out

    if hasattr(a, "get"):
        return a.get()
    try:
        return cp.asnumpy(a)
    except Exception:
        return a


# ============================================================
# Environment: geometría + comida (en GPU)
# ============================================================
class Environment:
    def __init__(self, r_max: float):
        self.r_max = float(r_max)

        # Hawk–Dove parameters
        self.V = float(P["FOOD_UNIT_ENERGY"])
        self.D = float(P["INJURY_COST_D"])
        self.T = float(P["TIME_COST_T"])

        # Food parameters
        self.food_max = int(P["FOOD_MAX"])
        self.food_spawn_rate = float(P["FOOD_SPAWN_RATE"])
        self.food_margin = float(P["FOOD_MARGIN"])
        self.food_spawn_r_min = float(P["FOOD_SPAWN_R_MIN"])
        self.food_ttl = int(P["FOOD_TTL"])
        self.grab_radius = float(P["GRAB_RADIUS"])

        # Food state (GPU arrays)
        self.food_x = cp.array([], dtype=cp.float32)
        self.food_y = cp.array([], dtype=cp.float32)
        self.food_age = cp.array([], dtype=cp.int32)  # steps since spawned

    @staticmethod
    def polar_to_cart(r, th):
        return r * cp.cos(th), r * cp.sin(th)

    @staticmethod
    def cart_to_polar(x, y):
        r = cp.sqrt(x * x + y * y)
        th = cp.arctan2(y, x) % (2 * cp.pi)
        return r, th

    def spawn_food(self, k: int):
        """Spawn uniforme en área, pero restringido a un anillo [r_min, r_max-margin]."""
        if k <= 0:
            return

        r_min = max(self.food_margin, self.food_spawn_r_min)
        r_max = self.r_max - self.food_margin
        if r_min >= r_max:
            return

        # Uniforme en área en un anillo: r = sqrt( u*(Rmax^2 - Rmin^2) + Rmin^2 )
        u = cp.random.random(k, dtype=cp.float32)
        r = cp.sqrt(u * (r_max * r_max - r_min * r_min) + r_min * r_min)
        th = cp.random.random(k, dtype=cp.float32) * (2 * cp.pi)

        x, y = self.polar_to_cart(r, th)
        self.food_x = cp.concatenate([self.food_x, x.astype(cp.float32)])
        self.food_y = cp.concatenate([self.food_y, y.astype(cp.float32)])
        self.food_age = cp.concatenate([self.food_age, cp.zeros(k, dtype=cp.int32)])

    def regen_food(self):
        """Regenera comida con Poisson, limitado por FOOD_MAX."""
        if int(self.food_x.size) >= self.food_max:
            return
        k = int(np.random.poisson(self.food_spawn_rate))  # Poisson CPU es suficiente
        k = min(k, self.food_max - int(self.food_x.size))
        self.spawn_food(k)

    def age_and_decay_food(self):
        """Incrementa edad y elimina comida vieja (TTL)."""
        if int(self.food_age.size) == 0:
            return
        self.food_age = self.food_age + 1
        keep = self.food_age <= self.food_ttl
        self.food_x = self.food_x[keep]
        self.food_y = self.food_y[keep]
        self.food_age = self.food_age[keep]

    def consume_food_indices(self, eaten_idx_cpu):
        """Elimina comida consumida (índices CPU)"""
        if not eaten_idx_cpu:
            return
        m = int(self.food_x.size)
        mask = np.ones(m, dtype=bool)
        mask[np.array(eaten_idx_cpu, dtype=int)] = False
        mask_gpu = cp.asarray(mask)
        self.food_x = self.food_x[mask_gpu]
        self.food_y = self.food_y[mask_gpu]
        self.food_age = self.food_age[mask_gpu]

    def nearest_food(self, cell_x, cell_y):
        """(idx, dist) por célula. O(N*M) pero GPU-friendly para tus tamaños."""
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
# Cell model: lógica de pagos/competencia (para extender luego)
# ============================================================
class Cell:
    """
    Modelo de comportamiento (NO crea un objeto por individuo).
    En el futuro aquí puedes meter cooperación, memoria, roles, etc.
    """
    HAWK = 1
    DOVE = 0

    @staticmethod
    def pair_contest(a_idx, b_idx, types_cpu, env: Environment):
        """
        Aplica la lógica Hawk–Dove del diagrama.
        Retorna: (winner_idx, injury_costs, time_costs)
        - injury_costs: dict idx->D (solo hawk-hawk perdedor)
        - time_costs: dict idx->T (solo dove-dove ambos)
        """
        a_h = types_cpu[a_idx] == Cell.HAWK
        b_h = types_cpu[b_idx] == Cell.HAWK

        injury = {}
        time = {}

        if a_h and b_h:
            # 50/50: uno gana, el otro paga lesión D
            if random.random() < 0.5:
                injury[b_idx] = injury.get(b_idx, 0.0) + env.D
                return a_idx, injury, time
            else:
                injury[a_idx] = injury.get(a_idx, 0.0) + env.D
                return b_idx, injury, time

        if a_h and (not b_h):
            return a_idx, injury, time

        if (not a_h) and b_h:
            return b_idx, injury, time

        # dove vs dove: ambos pagan tiempo T, winner 50/50
        time[a_idx] = time.get(a_idx, 0.0) + env.T
        time[b_idx] = time.get(b_idx, 0.0) + env.T
        return (a_idx if random.random() < 0.5 else b_idx), injury, time

    @staticmethod
    def tournament(contenders, types_cpu, env: Environment):
        """Torneo por pares: queda 1 campeón. Acumula costos por pelea."""
        contenders = contenders[:]
        random.shuffle(contenders)

        injury_costs = {}
        time_costs = {}

        while len(contenders) > 1:
            a = contenders.pop()
            b = contenders.pop()
            w, inj, tim = Cell.pair_contest(a, b, types_cpu, env)

            for k, v in inj.items():
                injury_costs[k] = injury_costs.get(k, 0.0) + v
            for k, v in tim.items():
                time_costs[k] = time_costs.get(k, 0.0) + v

            contenders.append(w)

        return contenders[0], injury_costs, time_costs


# ============================================================
# Population: estado vectorizado (GPU) + ledger de costos
# ============================================================
class Population:
    def __init__(self, env: Environment, cap: int):
        self.env = env
        self.cap = int(cap)

        # Estado
        self.x = cp.array([], dtype=cp.float32)
        self.y = cp.array([], dtype=cp.float32)
        self.energy = cp.array([], dtype=cp.float32)

        # type: 1=Hawk/Hack, 0=Dove/Dover
        self.type = cp.array([], dtype=cp.int8)

        # mode: 0=forage, 1=return
        self.mode = cp.array([], dtype=cp.int8)

        # home theta per cell (nido en el borde)
        self.home_th = cp.array([], dtype=cp.float32)

        # Ledger acumulado de costos por célula (para explicar muertes)
        self.cost_move_forage = cp.array([], dtype=cp.float32)
        self.cost_move_return = cp.array([], dtype=cp.float32)
        self.cost_randwalk = cp.array([], dtype=cp.float32)
        self.cost_fight_injury = cp.array([], dtype=cp.float32)
        self.cost_fight_time = cp.array([], dtype=cp.float32)
        self.cost_other = cp.array([], dtype=cp.float32)

    def size(self) -> int:
        return int(self.energy.size)

    def init_random(self, n0: int, hack_ratio: float):
        th = cp.random.random(n0, dtype=cp.float32) * (2 * cp.pi)
        r = cp.full(n0, self.env.r_max, dtype=cp.float32)
        x, y = self.env.polar_to_cart(r, th)

        e = cp.asarray(np.random.uniform(P["ENERGY_INIT_MIN"], P["ENERGY_INIT_MAX"], n0), dtype=cp.float32)
        t = cp.asarray((np.random.random(n0) < hack_ratio).astype(np.int8))
        mode = cp.zeros(n0, dtype=cp.int8)

        self.x, self.y = x.astype(cp.float32), y.astype(cp.float32)
        self.energy = e
        self.type = t
        self.mode = mode
        self.home_th = th.astype(cp.float32)

        z = cp.zeros(n0, dtype=cp.float32)
        self.cost_move_forage = z.copy()
        self.cost_move_return = z.copy()
        self.cost_randwalk = z.copy()
        self.cost_fight_injury = z.copy()
        self.cost_fight_time = z.copy()
        self.cost_other = z.copy()

    def _append_ledger(self, k):
        z = cp.zeros(k, dtype=cp.float32)
        self.cost_move_forage = cp.concatenate([self.cost_move_forage, z])
        self.cost_move_return = cp.concatenate([self.cost_move_return, z])
        self.cost_randwalk = cp.concatenate([self.cost_randwalk, z])
        self.cost_fight_injury = cp.concatenate([self.cost_fight_injury, z])
        self.cost_fight_time = cp.concatenate([self.cost_fight_time, z])
        self.cost_other = cp.concatenate([self.cost_other, z])

    def add_children(self, child_x, child_y, child_energy, child_type, child_home_th):
        if int(child_energy.size) == 0:
            return 0
        room = self.cap - self.size()
        if room <= 0:
            return 0
        k = int(min(room, int(child_energy.size)))

        self.x = cp.concatenate([self.x, child_x[:k]])
        self.y = cp.concatenate([self.y, child_y[:k]])
        self.energy = cp.concatenate([self.energy, child_energy[:k]])
        self.type = cp.concatenate([self.type, child_type[:k]])
        self.mode = cp.concatenate([self.mode, cp.zeros(k, dtype=cp.int8)])
        self.home_th = cp.concatenate([self.home_th, child_home_th[:k]])

        self._append_ledger(k)
        return k

    def kill_dead_and_classify(self, death_counters):
        """
        Elimina muertos y clasifica causa probable por 'mayor costo acumulado'.
        death_counters: dict que acumula muertes por causa.
        """
        dead = self.energy <= 0
        dead_idx = cp.where(dead)[0]
        dead_n = int(dead_idx.size)

        if dead_n > 0:
            # (dead_n, 5) para argmax
            costs = cp.stack([
                self.cost_move_forage[dead_idx],
                self.cost_move_return[dead_idx],
                self.cost_randwalk[dead_idx],
                self.cost_fight_injury[dead_idx],
                self.cost_fight_time[dead_idx],
            ], axis=1)

            cause = cp.argmax(costs, axis=1)  # 0..4
            cause_cpu = to_cpu(cause)

            names = ["travel_forage", "travel_return", "random_walk", "fight_injury", "fight_time"]
            for c in cause_cpu.tolist():
                death_counters[names[int(c)]] = death_counters.get(names[int(c)], 0) + 1

        # filtrar vivos
        alive = ~dead
        self.x = self.x[alive]
        self.y = self.y[alive]
        self.energy = self.energy[alive]
        self.type = self.type[alive]
        self.mode = self.mode[alive]
        self.home_th = self.home_th[alive]

        self.cost_move_forage = self.cost_move_forage[alive]
        self.cost_move_return = self.cost_move_return[alive]
        self.cost_randwalk = self.cost_randwalk[alive]
        self.cost_fight_injury = self.cost_fight_injury[alive]
        self.cost_fight_time = self.cost_fight_time[alive]
        self.cost_other = self.cost_other[alive]

        return dead_n


# ============================================================
# STEP de simulación (GPU + CPU solo para torneos por comida)
# ============================================================
def step_world(env: Environment, pop: Population, stats):
    births = 0

    # Food: regen + aging/decay (evita acumulación)
    env.regen_food()
    env.age_and_decay_food()

    n = pop.size()
    if n == 0:
        return 0, 0, 0

    # Clamp energía
    pop.energy = cp.minimum(pop.energy, cp.float32(P["ENERGY_CAP"]))

    # Polar
    r = cp.sqrt(pop.x * pop.x + pop.y * pop.y)
    at_nest = r >= (env.r_max - P["NEST_THICKNESS"])

    # Si energía baja -> return
    low = (pop.energy < cp.float32(P["RETURN_IF_LOW"])) & (pop.mode == 0)
    pop.mode = cp.where(low, cp.int8(1), pop.mode)

    # --------------------------------------------------------
    # Return movement (hacia home)
    # --------------------------------------------------------
    n = pop.size()
    home_r = cp.full(n, env.r_max, dtype=cp.float32)
    hx, hy = env.polar_to_cart(home_r, pop.home_th)

    dx = hx - pop.x
    dy = hy - pop.y
    dist = cp.sqrt(dx * dx + dy * dy) + 1e-9

    step = cp.minimum(cp.full(n, P["SPEED"], dtype=cp.float32), dist)
    ux = dx / dist
    uy = dy / dist

    noise = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.04
    cs = cp.cos(noise)
    sn = cp.sin(noise)
    ux2 = ux * cs - uy * sn
    uy2 = ux * sn + uy * cs

    is_return = (pop.mode == 1)
    pop.x = cp.where(is_return, pop.x + step * ux2, pop.x)
    pop.y = cp.where(is_return, pop.y + step * uy2, pop.y)

    cost_return = (cp.float32(P["MOVE_COST_BASE"]) + cp.float32(P["MOVE_COST_PER_STEP"]) * step)
    pop.energy = pop.energy - cp.where(is_return, cost_return, 0.0).astype(cp.float32)
    pop.cost_move_return = pop.cost_move_return + cp.where(is_return, cost_return, 0.0).astype(cp.float32)

    # Recalcular nido
    r = cp.sqrt(pop.x * pop.x + pop.y * pop.y)
    at_nest = r >= (env.r_max - P["NEST_THICKNESS"])

    # --------------------------------------------------------
    # Reproducción SOLO en nido (y solo si estaba en return)
    # --------------------------------------------------------
    is_return = (pop.mode == 1)
    repro = is_return & at_nest & (pop.energy >= cp.float32(P["REPRO_THRESHOLD"]))
    repro_idx = cp.where(repro)[0]

    if int(repro_idx.size) > 0:
        frac = float(P["REPRO_FRACTION"])
        child_energy = pop.energy[repro_idx] * cp.float32(frac)
        pop.energy[repro_idx] = pop.energy[repro_idx] * cp.float32(1.0 - frac)

        k = int(repro_idx.size)
        child_th = (pop.home_th[repro_idx] + (cp.random.random(k, dtype=cp.float32) - 0.5) * 0.28) % (2 * cp.pi)
        child_r = cp.full(k, env.r_max, dtype=cp.float32)
        child_x, child_y = env.polar_to_cart(child_r, child_th)

        child_type = pop.type[repro_idx]
        added = pop.add_children(child_x, child_y, child_energy, child_type, child_th)
        births += added

        # Si se añadieron hijos, recomputar tamaños/mascaras para evitar mismatches
        if added > 0:
            n = pop.size()
            r = cp.sqrt(pop.x * pop.x + pop.y * pop.y)
            at_nest = r >= (env.r_max - P["NEST_THICKNESS"])
            is_return = (pop.mode == 1)

    # Llegó al nido -> sale (mode=forage) + empujón hacia adentro
    arrived = is_return & at_nest
    pop.mode = cp.where(arrived, cp.int8(0), pop.mode)

    rr = cp.sqrt(pop.x * pop.x + pop.y * pop.y) + 1e-9
    target_r = cp.float32(env.r_max - 0.9)
    scale = target_r / rr
    pop.x = cp.where(arrived, pop.x * scale, pop.x)
    pop.y = cp.where(arrived, pop.y * scale, pop.y)

    # --------------------------------------------------------
    # Forage: nearest food pursuit or random walk
    # --------------------------------------------------------
    n = pop.size()
    is_forage = (pop.mode == 0)

    idx_food, dist_food = env.nearest_food(pop.x, pop.y)
    if idx_food is None:
        # random walk for foragers
        dth = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.44
        dr = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.30
        th = cp.arctan2(pop.y, pop.x) % (2 * cp.pi)
        r = cp.sqrt(pop.x * pop.x + pop.y * pop.y)
        th2 = (th + dth) % (2 * cp.pi)
        r2 = cp.clip(r + dr, 0.0, env.r_max)
        x2, y2 = env.polar_to_cart(r2, th2)

        pop.x = cp.where(is_forage, x2, pop.x)
        pop.y = cp.where(is_forage, y2, pop.y)
        pop.energy = pop.energy - cp.where(is_forage, cp.float32(P["RANDWALK_COST"]), 0.0)
        pop.cost_randwalk = pop.cost_randwalk + cp.where(is_forage, cp.float32(P["RANDWALK_COST"]), 0.0)
    else:
        detect = dist_food < cp.float32(P["SENSORY_RANGE"] * P["SENSORY_FACTOR"])
        pursue = is_forage & detect

        fx = env.food_x[idx_food]
        fy = env.food_y[idx_food]

        dx = fx - pop.x
        dy = fy - pop.y
        dist = cp.sqrt(dx * dx + dy * dy) + 1e-9

        step2 = cp.minimum(cp.full(n, P["SPEED"], dtype=cp.float32), dist)
        ux = dx / dist
        uy = dy / dist

        noise = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.12
        cs = cp.cos(noise)
        sn = cp.sin(noise)
        ux2 = ux * cs - uy * sn
        uy2 = ux * sn + uy * cs

        pop.x = cp.where(pursue, pop.x + step2 * ux2, pop.x)
        pop.y = cp.where(pursue, pop.y + step2 * uy2, pop.y)

        cost_forage = (cp.float32(P["MOVE_COST_BASE"]) + cp.float32(P["MOVE_COST_PER_STEP"]) * step2)
        pop.energy = pop.energy - cp.where(pursue, cost_forage, 0.0).astype(cp.float32)
        pop.cost_move_forage = pop.cost_move_forage + cp.where(pursue, cost_forage, 0.0).astype(cp.float32)

        # random walk if not pursuing
        wander = is_forage & (~detect)
        dth = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.44
        dr = (cp.random.random(n, dtype=cp.float32) - 0.5) * 0.30
        th = cp.arctan2(pop.y, pop.x) % (2 * cp.pi)
        r = cp.sqrt(pop.x * pop.x + pop.y * pop.y)
        th2 = (th + dth) % (2 * cp.pi)
        r2 = cp.clip(r + dr, 0.0, env.r_max)
        x2, y2 = env.polar_to_cart(r2, th2)

        pop.x = cp.where(wander, x2, pop.x)
        pop.y = cp.where(wander, y2, pop.y)
        pop.energy = pop.energy - cp.where(wander, cp.float32(P["RANDWALK_COST"]), 0.0)
        pop.cost_randwalk = pop.cost_randwalk + cp.where(wander, cp.float32(P["RANDWALK_COST"]), 0.0)

        # candidates: within grab radius
        dx2 = fx - pop.x
        dy2 = fy - pop.y
        dist2 = cp.sqrt(dx2 * dx2 + dy2 * dy2)
        can_grab = pursue & (dist2 < cp.float32(env.grab_radius))

        cand_cells = to_cpu(cp.where(can_grab)[0])
        cand_food = to_cpu(idx_food[can_grab])

        # Bucket by food idx (CPU, legible)
        buckets = {}
        for ci, fj in zip(cand_cells.tolist(), cand_food.tolist()):
            buckets.setdefault(int(fj), []).append(int(ci))

        eaten_idx = []
        if buckets:
            types_cpu = to_cpu(pop.type).astype(int)

            injury_delta = {}
            time_delta = {}
            winners = []

            for food_j, group in buckets.items():
                champ, inj, tim = Cell.tournament(group, types_cpu, env)
                winners.append(champ)
                eaten_idx.append(food_j)

                for k, v in inj.items():
                    injury_delta[k] = injury_delta.get(k, 0.0) + v
                for k, v in tim.items():
                    time_delta[k] = time_delta.get(k, 0.0) + v

            # apply fight costs (GPU)
            if injury_delta:
                idxs = cp.asarray(list(injury_delta.keys()), dtype=cp.int32)
                dvs = cp.asarray(list(injury_delta.values()), dtype=cp.float32)
                pop.energy[idxs] = pop.energy[idxs] - dvs
                pop.cost_fight_injury[idxs] = pop.cost_fight_injury[idxs] + dvs

            if time_delta:
                idxs = cp.asarray(list(time_delta.keys()), dtype=cp.int32)
                dvs = cp.asarray(list(time_delta.values()), dtype=cp.float32)
                pop.energy[idxs] = pop.energy[idxs] - dvs
                pop.cost_fight_time[idxs] = pop.cost_fight_time[idxs] + dvs

            # winners: +V and return
            if winners:
                win_idx = cp.asarray(winners, dtype=cp.int32)
                pop.energy[win_idx] = pop.energy[win_idx] + cp.float32(env.V)
                pop.mode[win_idx] = cp.int8(1)

            # consume food
            env.consume_food_indices(eaten_idx)

            # “despegar” perdedores (barato)
            winners_set = set(winners)
            losers = [c for group in buckets.values() for c in group if c not in winners_set]
            if losers:
                losers_idx = cp.asarray(losers, dtype=cp.int32)
                ang = cp.asarray(np.random.uniform(0.9, 1.6, len(losers)), dtype=cp.float32)
                cs = cp.cos(ang)
                sn = cp.sin(ang)
                x0 = pop.x[losers_idx]
                y0 = pop.y[losers_idx]
                pop.x[losers_idx] = x0 * cs - y0 * sn
                pop.y[losers_idx] = x0 * sn + y0 * cs
                pop.energy[losers_idx] = pop.energy[losers_idx] - cp.float32(0.15)
                pop.cost_other[losers_idx] = pop.cost_other[losers_idx] + cp.float32(0.15)

    # --------------------------------------------------------
    # Kill + classify deaths
    # --------------------------------------------------------
    deaths = pop.kill_dead_and_classify(stats["death_causes"])

    stats["births"] += births
    stats["deaths"] += deaths

    return births, deaths, int(env.food_x.size)


# ============================================================
# Setup
# ============================================================
random.seed(2)
np.random.seed(2)

env = Environment(P["R_MAX"])
env.spawn_food(90)

pop = Population(env, P["POP_CAP"])
pop.init_random(P["N0"], P["HACK_RATIO0"])

stats = dict(
    turn=0,
    births=0,
    deaths=0,
    death_causes={},  # counts by cause
)

# ============================================================
# Plot: panel stats + slider speed
# ============================================================
fig = plt.figure(figsize=(10.2, 7.2), facecolor="black")

ax_stats = fig.add_axes([0.02, 0.10, 0.30, 0.84], facecolor="black")
ax_stats.set_axis_off()
stats_txt = ax_stats.text(
    0.02, 0.98, "", va="top", ha="left",
    color="white", fontsize=10, family="monospace"
)

ax_slider = fig.add_axes([0.05, 0.04, 0.25, 0.03], facecolor="black")
speed_slider = Slider(ax_slider, "steps/frame", valmin=1, valmax=50, valinit=5, valstep=1)
speed_slider.label.set_color("white")
speed_slider.valtext.set_color("white")

ax = fig.add_axes([0.35, 0.06, 0.63, 0.88], projection="polar", facecolor="black")
ax.set_ylim(0, env.r_max)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["polar"].set_edgecolor("white")
ax.spines["polar"].set_linewidth(1.6)
ax.set_title("Hack vs Dover (CuPy)", color="white", pad=18)

ring_th = np.linspace(0, TWOPI, 800)
ax.plot(ring_th, np.full_like(ring_th, env.r_max), color="white", linewidth=2.0, alpha=0.95)

# scatters
food_r0, food_th0 = to_cpu(env.cart_to_polar(env.food_x, env.food_y))
food_sc = ax.scatter(food_th0, food_r0, s=34, c="green", alpha=0.65)

r0, th0 = to_cpu(env.cart_to_polar(pop.x, pop.y))
types0 = to_cpu(pop.type)
colors0 = np.array(["red" if t == 1 else "dodgerblue" for t in types0])
sizes0 = np.clip(to_cpu(pop.energy), 10, 200)
cell_sc = ax.scatter(th0, r0, s=sizes0, c=colors0, alpha=0.85)


def format_death_causes(dc: dict):
    total = sum(dc.values()) if dc else 0
    if total == 0:
        return "death causes: (no deaths yet)"
    order = sorted(dc.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["death causes (share):"]
    for k, v in order[:5]:
        lines.append(f"  {k:13s}: {v:5d} ({100.0*v/total:5.1f}%)")
    return "\n".join(lines)


def animate(_):
    steps = int(speed_slider.val)

    births_acc = 0
    deaths_acc = 0
    food_n = int(env.food_x.size)

    for _ in range(steps):
        stats["turn"] += 1
        b, d, food_n = step_world(env, pop, stats)
        births_acc += b
        deaths_acc += d
        if pop.size() == 0:
            break

    # Update food scatter
    if int(env.food_x.size) > 0:
        fr, fth = env.cart_to_polar(env.food_x, env.food_y)
        food_sc.set_offsets(np.c_[to_cpu(fth), to_cpu(fr)])
        food_sc.set_sizes(np.full(int(env.food_x.size), 34.0))
    else:
        food_sc.set_offsets(np.empty((0, 2)))

    # Update cell scatter
    if pop.size() == 0:
        cell_sc.set_offsets(np.empty((0, 2)))
        stats_txt.set_text("Extinción total.\n")
        return food_sc, cell_sc, stats_txt

    r, th = env.cart_to_polar(pop.x, pop.y)
    r_cpu = to_cpu(r)
    th_cpu = to_cpu(th)

    energies_cpu = to_cpu(pop.energy)
    types_cpu = to_cpu(pop.type)

    colors = np.array(["red" if t == 1 else "dodgerblue" for t in types_cpu])
    sizes = np.clip(energies_cpu, 10, 200)

    cell_sc.set_offsets(np.c_[th_cpu, r_cpu])
    cell_sc.set_sizes(sizes)
    cell_sc.set_color(colors)

    total = pop.size()
    hawks = int((types_cpu == 1).sum())
    doves = total - hawks
    ph = 100.0 * hawks / total
    pd = 100.0 * doves / total

    # Promedios de costos acumulados (vivos)
    cmf = float(to_cpu(cp.mean(pop.cost_move_forage)))
    cmr = float(to_cpu(cp.mean(pop.cost_move_return)))
    crw = float(to_cpu(cp.mean(pop.cost_randwalk)))
    cfi = float(to_cpu(cp.mean(pop.cost_fight_injury)))
    cft = float(to_cpu(cp.mean(pop.cost_fight_time)))

    stats_txt.set_text(
        f"steps/frame: {steps}\n"
        f"turn: {stats['turn']}\n"
        f"pop : {total:4d}   births(step): {births_acc:3d}   deaths(step): {deaths_acc:3d}\n"
        f"Hawk: {hawks:4d} ({ph:5.1f}%)   Dove: {doves:4d} ({pd:5.1f}%)\n"
        f"E(mean/med): {energies_cpu.mean():7.2f} / {np.median(energies_cpu):7.2f}\n"
        f"food active: {food_n:4d}\n"
        f"\n"
        f"HAWK–DOVE\n"
        f"V={env.V:.1f}  D={env.D:.1f}  T={env.T:.1f}\n"
        f"\n"
        f"mean accumulated costs (alive)\n"
        f"  move_forage : {cmf:8.2f}\n"
        f"  move_return : {cmr:8.2f}\n"
        f"  rand_walk   : {crw:8.2f}\n"
        f"  fight_injury: {cfi:8.2f}\n"
        f"  fight_time  : {cft:8.2f}\n"
        f"\n"
        f"{format_death_causes(stats['death_causes'])}\n"
        f"\n"
        f"FOOD\n"
        f"spawn_rate={P['FOOD_SPAWN_RATE']:.1f}/step  max={P['FOOD_MAX']}\n"
        f"ttl={P['FOOD_TTL']}  r_min={P['FOOD_SPAWN_R_MIN']:.1f}\n"
        f"grab_r={P['GRAB_RADIUS']:.2f}\n"
        f"\n"
        f"REPRO\n"
        f"thr={P['REPRO_THRESHOLD']:.1f}  frac={P['REPRO_FRACTION']:.2f}\n"
    )

    return food_sc, cell_sc, stats_txt


ani = FuncAnimation(fig, animate, frames=1200, interval=45, blit=False)
plt.show()
