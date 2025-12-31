from Classes import *
from Utilities import *
from Simulation import *
from Visualization import *

# ----------------------------
# Parameters
# ----------------------------
#Universe parameters
NUM_CYCLES = 500
STEP_PER_CYCLE = 100

GRID_SIZE_X = 1600
GRID_SIZE_Y = 900

NEST_RADIUS = 5.0
WATER_SOURCES_RADIUS = 10.0

# Caracteristics of cell parameters: Cost of being alive and moving
CELL_METABOLISM_BASE = 0.020
STEP_SIZE = 1.0                  # Distance moved per step
REPRO_ENERGY_THRESHOLD = 30.0
VELOCITY_COST_FACTOR = 0.010
THIRST_PENALTY = 0.020
HYDRATION_COST_FACTOR = 0.012


# Iteraction with universe parameters
FOOD_VALUE = 10.0
INITIAL_ENERGY = 20.0
INITIAL_POPULATION = 20
INITIAL_HYDRATION = 10.0

# Intercation between cells parameters: Costo of being with other cells
# (Not implemented in this version)
# (Payment Matrix: GAME THEORY and EVOLUTIONARY STABLE STRATEGIES)

# ----------------------------
# Main Simulation Loop
# ----------------------------
if __name__ == "__main__":
    # World
    u = Universe(size_x=100, size_y=100, food_value=7.5, nest_radius=8, water_sources_radius=10)
    u.gen_water_sources()
    u.spawn_food(120)

    # Population
    cells = [Cell(np.random.uniform(0, 100), np.random.uniform(0, 100), energy=18, hydration=12) for _ in range(40)]
    pop = Population(cells, u)

    # Run
    Simulation(u, pop).animate(steps=4000, food_spawn_per_step=2, max_food=260, interval_ms=16)