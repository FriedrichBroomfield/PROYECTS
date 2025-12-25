import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Hack vs Dover

class Environment:  # Polar
    def __init__(self, r_min=0, r_max=10, nr=200, ntheta=360):
        self.r_min = r_min
        self.r_max = r_max
        self.nr = nr
        self.ntheta = ntheta
        self.r = np.linspace(r_min, r_max, nr)
        self.theta = np.linspace(0, 2 * np.pi, ntheta)
        self.Theta, self.R = np.meshgrid(self.theta, self.r)

    def food_distribution(self):
        return np.ones((self.nr, self.ntheta))


class Cell:  # Two Kinds of Cells: Hacks and Dovers
    def __init__(self, r, theta):
        # Start at nest (outer ring). home stores nest location.
        self.home_r = r
        self.home_theta = theta
        self.r = r
        self.theta = theta
        self.sensory_range = 1.0
        self.cell_type = "Dover"  # For now, all Dovers; will randomize later
        # self.cell_type = random.choice(['Hack', 'Dover'])
        self.energy = 100

    def move(self, dr, dtheta):
        self.r = np.clip(self.r + dr, 0, env.r_max)
        self.theta = (self.theta + dtheta) % (2 * np.pi)

    def get_position(self):
        return (self.r, self.theta)

    def fight_for_food(self, other):
        # Adjust energies according to types
        if self.cell_type == 'Hack' and other.cell_type == 'Dover':
            self.energy += 50
            other.energy -= 30
        elif self.cell_type == 'Dover' and other.cell_type == 'Hack':
            self.energy -= 30
            other.energy += 50
        elif self.cell_type == other.cell_type == 'Hack':
            self.energy -= 20
            other.energy -= 20
        else:
            # same non-Hack type or mixed other cases
            self.energy = 25
            other.energy = 25

    def reproduce(self):
        # Reproduce only at nest after returning
        if self.energy > 150:
            self.energy /= 2
            child_theta = (self.home_theta + random.uniform(-0.1, 0.1)) % (2 * np.pi)
            child = Cell(self.home_r, child_theta)
            child.cell_type = self.cell_type
            child.energy = self.energy
            return child
        return None
    def search_for_food(self): 
    


# --- Simulation setup and animation ---
env = Environment(r_min=0, r_max=10)
N = 400
cells = [Cell(env.r_max, random.uniform(0, 2 * np.pi)) for _ in range(N)]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='polar')


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# persistent food sources for smoother, meaningful contests
np.random.seed(1)
n_food = 60
food_r = np.random.uniform(env.r_min + 0.5, env.r_max - 1.0, n_food)
food_theta = np.random.uniform(0, 2 * np.pi, n_food)
food_amount = np.full(n_food, 50.0)


def frame(i):
    ax.clear()
    ax.set_ylim(0, env.r_max)
    ax.set_title(f"Turn {i}")

    # draw food (size ∝ remaining amount)
    ax.scatter(food_theta, food_r, c='green', s=np.clip(food_amount, 1, 200), alpha=0.6)

    # Each cell leaves nest, forages inward toward a target food, then returns to nest.
    # We'll compute temporary forage positions to handle contests/consumption, then restore nest positions.
    fx = food_r * np.cos(food_theta)
    fy = food_r * np.sin(food_theta)

    # assign nearest food target for each cell (based at nest)
    targets = {fi: [] for fi in range(n_food)}
    for c in cells:
        # choose nearest food from home position
        cx, cy = polar_to_cartesian(c.home_r, c.home_theta)
        dists = np.hypot(fx - cx, fy - cy)
        fi = int(np.argmin(dists))
        if food_amount[fi] > 0:
            # compute a temporary foraging position (inward from nest toward food)
            dir_x, dir_y = fx[fi] - cx, fy[fi] - cy
            norm = np.hypot(dir_x, dir_y)
            if norm < 1e-6:
                # already on top (unlikely) -> small inward offset
                forage_r = max(env.r_min + 0.1, c.home_r - 1.0)
                forage_theta = c.home_theta
            else:
                # choose foraging depth: they go out (inward) some random fraction each turn
                frac = random.uniform(0.25, 0.6)
                nx = cx + frac * dir_x
                ny = cy + frac * dir_y
                forage_r = np.hypot(nx, ny)
                forage_theta = np.arctan2(ny, nx) % (2 * np.pi)
            # cap within arena and within sensory reach for contest logic
            forage_r = np.clip(forage_r, env.r_min, env.r_max)
            # store temporary attributes on the cell for this turn
            c._forage_r = forage_r
            c._forage_theta = forage_theta
            c._forage_xy = polar_to_cartesian(forage_r, forage_theta)
            targets[fi].append(c)
        else:
            # no food target (depleted) -> small inward wander, but still return to nest
            c._forage_r = max(env.r_min + 0.1, c.home_r - random.uniform(0.2, 0.6))
            c._forage_theta = (c.home_theta + random.uniform(-0.2, 0.2)) % (2 * np.pi)
            c._forage_xy = polar_to_cartesian(c._forage_r, c._forage_theta)

    # Fights during the forage trip: if multiple cells visit same food and are mutually close in the forage positions
    for fi, clist in targets.items():
        if len(clist) <= 1 or food_amount[fi] <= 0:
            continue
        for a_idx in range(len(clist)):
            for b_idx in range(a_idx + 1, len(clist)):
                a = clist[a_idx]
                b = clist[b_idx]
                ax_a, ay_a = a._forage_xy
                ax_b, ay_b = b._forage_xy
                if np.hypot(ax_a - ax_b, ay_a - ay_b) <= max(a.sensory_range, b.sensory_range):
                    a.fight_for_food(b)

    # Consumption happens at forage positions
    for fi, clist in targets.items():
        if food_amount[fi] <= 0:
            continue
        for c in clist:
            cx, cy = c._forage_xy
            if np.hypot(cx - fx[fi], cy - fy[fi]) < 0.6:
                gain = min(30, food_amount[fi])
                c.energy += gain
                food_amount[fi] -= gain

    # Travel costs and return to nest: subtract small energy for the trip and reset position to home
    for c in cells:
        trip_cost = 1 + 0.05 * abs(c.home_r - c._forage_r)  # small cost proportional to travel
        c.energy -= trip_cost
        # Slight drift in home_theta to simulate settling; nest remains outer ring
        c.home_theta = (c.home_theta + random.uniform(-0.02, 0.02)) % (2 * np.pi)
        c.r = c.home_r
        c.theta = c.home_theta
        # cleanup temporary forage attributes
        del c._forage_r, c._forage_theta, c._forage_xy

    # Reproduction and cleanup at nest
    new_cells = []
    survivors = []
    for c in cells:
        if c.energy <= 0:
            continue
        child = c.reproduce()
        if child:
            new_cells.append(child)
        survivors.append(c)
    cells[:] = survivors + new_cells

    # Plot cells (polar scatter: theta, r) at nest (outer ring)
    rs = np.array([c.r for c in cells])
    thetas = np.array([c.theta for c in cells])
    colors = ['red' if c.cell_type == 'Hack' else 'blue' for c in cells]
    sizes = np.clip([c.energy for c in cells], 10, 200)
    ax.scatter(thetas, rs, c=colors, s=sizes, alpha=0.8)


ani = FuncAnimation(fig, frame, frames=500, interval=100)
plt.show()

        