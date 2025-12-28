from Parametros import *

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

        # la condición esencial: energía >= threshold
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
# Simulation (global day/night)
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

        # push inside a bit
        r = cp.sqrt(self.pop.x * self.pop.x + self.pop.y * self.pop.y) + 1e-9
        target_r = cp.float32(self.env.r_max - 0.9)
        scale = target_r / r
        self.pop.x *= scale
        self.pop.y *= scale

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
        if self.pop.size() == 0:
            return

        if self.phase == "day":
            self.step_day()
            self.phase_step += 1
            if self.phase_step >= P["DAY_STEPS"]:
                self.start_night()
        else:
            self.step_night()
            self.phase_step += 1
            if self.phase_step >= P["NIGHT_STEPS"]:
                self.start_day()

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

        if int(self.env.food_x.size) == 0:
            d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
            self.stats["deaths_today"] += d
            return

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
            # no food -> brownian for seekers
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

        # grab candidates: only those still below SATIETY_E compete for food
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

            # apply fight costs
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

            # consume food
            self.env.consume_food_indices(eaten_idx)

        # daily priority 2: near end of day, if safe, stop and return
        remaining = (P["DAY_STEPS"] - (self.phase_step + 1))
        if remaining <= int(0.25 * P["DAY_STEPS"]):
            safe = (self.pop.energy >= cp.float32(P["SURVIVE_GOAL"]))
            can_stop = (~self.pop.done_for_day) & (~self.pop.ready_to_repro) & safe
            self.pop.done_for_day |= can_stop

        d = self.pop.kill_dead_and_classify(self.stats["death_causes"])
        self.stats["deaths_today"] += d