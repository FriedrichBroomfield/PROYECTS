from Utilities import *
from Parameters import *




# ----------------------------
# World
# ----------------------------
class Universe:
    def __init__(self, size_x=GRID_SIZE_X, size_y=GRID_SIZE_Y, food_value=FOOD_VALUE, nest_radius=NEST_RADIUS, water_sources_radius=WATER_SOURCES_RADIUS):
        
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.food_value = float(food_value)
        self.nest_radius = float(nest_radius)
        self.water_sources_radius = float(water_sources_radius)

        self.food_x = np.empty(0, dtype=float)
        self.food_y = np.empty(0, dtype=float)
        self.water_sources = np.empty((0, 2), dtype=float)

    def gen_water_sources(self):
        
        """ Somewhat evenly distribute water sources in the world and away from the center/nest as randomly possible.
        * Water sources are circular areas where cells can drink to restore hydration.
        * They should be sufficiently apart from each other and from the nest area.
        * They should wrap around the torus world: if in one edge, consider the opposite edge drinkable too, as plotable.
        * evenly distribute n_sources water sources means: maximize distances between them.
        
        
        """
        n_sources = 4
        positions = []
        attempts, max_attempts = 0, 1000
        while len(positions) < n_sources and attempts < max_attempts:
            x = np.random.uniform(0, self.size_x)
            y = np.random.uniform(0, self.size_y)

            # keep away from nest center
            cx, cy = self.size_x / 2, self.size_y / 2
            dx = torus_delta(x, cx, self.size_x)
            dy = torus_delta(y, cy, self.size_y)
            if (dx * dx + dy * dy) < (self.nest_radius + self.water_sources_radius) ** 2:
                attempts += 1
                continue

            # keep away from other water sources
            min_sep2 = (self.water_sources_radius * 2.5) ** 2
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
        self.water_sources = np.array(positions, dtype=float)

    def spawn_food(self, n_new: int):
        """Append n_new food particles with simple collision checks."""
        if n_new <= 0:
            return

        fx, fy = [], []
        attempts, max_attempts = 0, max(200, n_new * 30)

        # Existing food to avoid overlap
        existing_x = self.food_x
        existing_y = self.food_y

        while len(fx) < n_new and attempts < max_attempts:
            x = np.random.uniform(0, self.size_x)
            y = np.random.uniform(0, self.size_y)

            # keep away from water blobs
            if self.water_sources.size:
                d2w = torus_dist2(self.water_sources[:, 0], self.water_sources[:, 1], x, y, self.size_x, self.size_y)
                if np.any(d2w < (self.water_sources_radius ** 2)):
                    attempts += 1
                    continue

            # avoid overlapping food (soft)
            min_sep2 = 1.0 ** 2
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
            self.food_x = np.concatenate([self.food_x, np.array(fx)])
            self.food_y = np.concatenate([self.food_y, np.array(fy)])

    def remove_food_index(self, idx: int):
        self.food_x = np.delete(self.food_x, idx)
        self.food_y = np.delete(self.food_y, idx)


# ----------------------------
# Agent
# ----------------------------
class Cell:
    def __init__(self, x, y, energy=18.0, hydration=12.0,step_size=STEP_SIZE):
        self.x = float(x)
        self.y = float(y)

        # Smooth motion state: velocity (vx, vy) [Review later]
        self.vx = np.random.uniform(-0.4, 0.4)
        self.vy = np.random.uniform(-0.4, 0.4)

        # Resources
        self.energy = float(energy)
        self.hydration = float(hydration)

        self.alive = True
        self.thirsty = self.hydration < 6.0
        self.hungry = self.energy < 10.0
        
        
        self.repro_cooldown = 0
        self.step = step_size

    def decide_target(self, universe: Universe):
        """
        Returns a (tx, ty) steering vector in torus coordinates.
        Priority: water if very thirsty, food if hungry, otherwise wander.
        """
        thirsty = self.hydration < 6.0
        hungry = self.energy < 10.0

        if thirsty and universe.water_sources.size:
            # nearest water
            wx = universe.water_sources[:, 0]
            wy = universe.water_sources[:, 1]
            d2 = torus_dist2(wx, wy, self.x, self.y, universe.size_x, universe.size_y)
            i = int(np.argmin(d2))
            tx = torus_delta(wx[i], self.x, universe.size_x)
            ty = torus_delta(wy[i], self.y, universe.size_y)       
            return tx, ty

        if hungry and universe.food_x.size:
            # nearest food
            d2 = torus_dist2(universe.food_x, universe.food_y, self.x, self.y, universe.size_x, universe.size_y)
            i = int(np.argmin(d2))
            tx = torus_delta(universe.food_x[i], self.x, universe.size_x)
            ty = torus_delta(universe.food_y[i], self.y, universe.size_y)
            return tx, ty

        # gentle “wander”
        angle = np.random.uniform(0, 2 * np.pi)
        return np.cos(angle), np.sin(angle)

    def move(self, universe: Universe, max_speed=MAX_SPEED, accel=ACCELERATION, damping=DAMPING, noise=NOISE_RW, water_radius=WATER_SOURCES_RADIUS) -> float:
        
        #Target
        tx, ty = self.decide_target(universe)
        norm = float(np.hypot(tx, ty) + ALMOST_ZERO)
        tx, ty = tx / norm, ty / norm
        
        # Repel from nearest water source if any exist; otherwise no repulsion
        if universe.water_sources.size:
            wx = universe.water_sources[:, 0]
            wy = universe.water_sources[:, 1]
            d2 = torus_dist2(wx, wy, self.x, self.y, universe.size_x, universe.size_y)
            
            
        # use argmin safely because d2 is non-empty here
        i = int(np.argmin(d2))
        rdx = torus_delta(wx[i], self.x, universe.size_x)
        rdy = torus_delta(wy[i], self.y, universe.size_y)

       
        if d2[i]<DAMPING * water_radius: 
            # Repelling Water sources (quadratic falloff)
            rx = -10 * (rdx ** 2) + (DAMPING * water_radius) ** 2 * 10
            ry = -10 * (rdy ** 2) + (DAMPING * water_radius) ** 2 * 10
        else:
            rx = 0
            ry = 0
            

        # acceleration + noise + damping (smooth): accel*tz is the directed force
        # Separate the "intentional" velocity (base_v) from the repulsion (rx,ry).
        # Energy cost will be computed from base_v so repulsion does not increase metabolic cost.
        noise_x = np.random.normal(0, noise)
        noise_y = np.random.normal(0, noise)
        base_vx = damping * self.vx + accel * tx + noise_x
        base_vy = damping * self.vy + accel * ty + noise_y

        # cap repulsion magnitude to avoid huge instantaneous jumps
        max_repulse = 0.6
        rx = float(np.clip(rx, -max_repulse, max_repulse))
        ry = float(np.clip(ry, -max_repulse, max_repulse))

        # full velocity includes repulsion (affects movement), but energy cost uses base_v
        self.vx = base_vx + rx
        self.vy = base_vy + ry

        # speed used for metabolism excludes the repulsion component
        speed = float(np.hypot(base_vx, base_vy) + ALMOST_ZERO)
        
        # clamp speed
        if speed > max_speed:
            s = max_speed / speed
            self.vx *= s
            self.vy *= s

        # move + wrap around torus
        self.x = (self.x + self.vx) % universe.size_x
        self.y = (self.y + self.vy) % universe.size_y
         
        #[Later make sure thar water repel cell]  

        return speed

    def try_eat(self, universe: Universe, dist=1) -> bool:
        if not universe.food_x.size:
            return False
        d2 = torus_dist2(universe.food_x, universe.food_y, self.x, self.y, universe.size_x, universe.size_y)
        i = int(np.argmin(d2))
        if d2[i] <= dist * dist:
            self.energy += universe.food_value
            universe.remove_food_index(i)
            return True
        return False

    def try_drink(self, universe: Universe, dist=2.3) -> bool:
        
        #Check for water
        if not universe.water_sources.size:
            return False
        
        #Retrive water sources
        wx = universe.water_sources[:, 0]
        wy = universe.water_sources[:, 1]
        
        #Check distances
        d2 = torus_dist2(wx, wy, self.x, self.y, universe.size_x, universe.size_y)
        
        #Drink from the nearst water
        if float(np.min(d2)) <= dist * dist:
            self.hydration = min(20.0, self.hydration + 5.0)
            return True
        
        return False

    def in_nest(self, universe: Universe) -> bool:
        # As the nest is in the center, is the same as being in the half of every axes
        cx, cy = universe.size_x / 2, universe.size_y / 2
        dx = torus_delta(cx, self.x, universe.size_x)
        dy = torus_delta(cy, self.y, universe.size_y)
        return (dx * dx + dy * dy) <= universe.nest_radius ** 2

    def try_reproduce(self, universe: Universe, repro_energy=26.0, repro_hydration=10.0):
        if self.repro_cooldown > 0:
            return None
        if self.energy < repro_energy or self.hydration < repro_hydration:
            return None
        if not self.in_nest(universe):
            return None

        # Split resources, apply cooldown
        self.repro_cooldown = 220
        child_energy = self.energy * 0.45
        child_hyd = self.hydration * 0.45
        self.energy *= 0.55
        self.hydration *= 0.55

        # Small offset
        ox, oy = np.random.uniform(-1.0, 1.0, size=2)
        
        return Cell((self.x + ox) % universe.size_x, (self.y + oy) % universe.size_y, child_energy, child_hyd)

    def metabolize(self, speed: float):
        # Base metabolism + movement cost + thirst penalty
        self.hydration -= 0.035*0 + 0.010 * speed
        self.energy -= 0.020 + 0.012 * speed + (0.020 if self.hydration < 3.0 else 0.0)

        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

        if self.energy <= 0.0 or self.hydration <= 0.0:
            self.alive = False


# ----------------------------
# Population
# ----------------------------
class Population:
    def __init__(self, cells, universe: Universe):
        self.cells = list(cells)
        self.universe = universe

        self.births = 0
        self.deaths = 0
        self.food_eaten = 0

    def step(self):
        new_cells = []
        deaths_this_step = 0
        births_this_step = 0
        food_eaten_this_step = 0

        for c in self.cells:
            if not c.alive:
                continue

            speed = c.move(self.universe)
            
            if c.try_eat(self.universe):
                food_eaten_this_step += 1
            c.try_drink(self.universe)

            child = c.try_reproduce(self.universe)
            if child is not None:
                new_cells.append(child)
                births_this_step += 1

            c.metabolize(speed)
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

    def positions_energy(self):
        if not self.cells:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        x = np.array([c.x for c in self.cells], dtype=float)
        y = np.array([c.y for c in self.cells], dtype=float)
        e = np.array([c.energy for c in self.cells], dtype=float)
        h = np.array([c.hydration for c in self.cells], dtype=float)
        return x, y, e, h

# ----------------------------
# Simulation + Modern-ish Viz
# ----------------------------
class Simulation:
    def __init__(self, universe: Universe, population: Population):
        self.universe = universe
        self.population = population
        self._ani = None  # keep reference so animation doesn't get garbage-collected

    def animate(
        self,
        steps=3000,
        food_spawn_per_step=1,
        max_food=250,
        interval_ms=16,
    ):
        # Apply dark style BEFORE figure creation
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

        # Subtle star field (static)
        stars = np.random.uniform([0, 0], [self.universe.size_x, self.universe.size_y], size=(350, 2))
        ax.scatter(stars[:, 0], stars[:, 1], s=2, alpha=0.15)

        # Water sources (glowy blobs)
        for wx, wy in self.universe.water_sources:
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius, alpha=0.20))
            ax.add_patch(plt.Circle((wx, wy), self.universe.water_sources_radius * 0.55, alpha=0.28))

        # Nest ring
        cx, cy = self.universe.size_x / 2, self.universe.size_y / 2
        ax.add_patch(plt.Circle((cx, cy), self.universe.nest_radius, fill=False, linestyle="--", linewidth=1.2, alpha=0.6))

        # Food + cells
        food_sc = ax.scatter([], [], s=18, alpha=0.9)  # default color in dark_background is bright; ok

        # Cells: glow layer + main layer. Color by energy.
        cell_glow = ax.scatter([], [], s=[], alpha=0.10, cmap="plasma", vmin=0, vmax=20)
        cell_sc = ax.scatter([], [], s=[], alpha=0.95, cmap="plasma", vmin=0, vmax=20)

        hud = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            alpha=0.95
        )

        def init():
            # Draw initial state so the window isn't "empty" before the first update()
            # Food
            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            # Cells
            x, y, e, h = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])
                sizes = 14 + 2.0 * np.sqrt(np.clip(e, 0, 50)) + 1.2 * np.sqrt(np.clip(h, 0, 25))
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

        def update(_frame):
            # Environment updates
            if self.universe.food_x.size < max_food:
                self.universe.spawn_food(food_spawn_per_step)

            # Sim step
            stats = self.population.step()

            # Food positions
            if self.universe.food_x.size:
                food_sc.set_offsets(np.column_stack([self.universe.food_x, self.universe.food_y]))
            else:
                food_sc.set_offsets(np.empty((0, 2)))

            # Cells positions + styling
            x, y, e, h = self.population.positions_energy()
            if x.size:
                xy = np.column_stack([x, y])

                # Size responds to energy + hydration (looks “alive”)
                sizes = 14 + 2.0 * np.sqrt(np.clip(e, 0, 50)) + 1.2 * np.sqrt(np.clip(h, 0, 25))

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

