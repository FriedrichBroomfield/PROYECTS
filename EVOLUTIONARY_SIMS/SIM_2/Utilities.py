import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Universe parameters
NUM_CYCLES = 500
STEP_PER_CYCLE = 100

GRID_SIZE_X = 100
GRID_SIZE_Y = 100

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


def torus_delta(a: np.ndarray, b: float, size: float) -> np.ndarray:
    """
    Smallest signed displacement from b -> a on a 1D torus of length=size.
    Returns values in [-size/2, size/2).
    """
    return (a - b + size / 2) % size - size / 2

def torus_dist2(ax: np.ndarray, ay: np.ndarray, bx: float, by: float, sx: float, sy: float) -> np.ndarray:
    dx = torus_delta(ax, bx, sx)
    dy = torus_delta(ay, by, sy)
    return dx * dx + dy * dy

def move_to(pos_x: np.ndarray, pos_y: np.ndarray, target_x: float, target_y: float, max_step: float, size_x: float, size_y: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Move from (pos_x, pos_y) towards (target_x, target_y) by up to max_step on a 2D torus of size (size_x, size_y).
    Returns new (pos_x, pos_y).
    """
    #Delta vectors
    dx = torus_delta(target_x, pos_x, size_x)
    dy = torus_delta(target_y, pos_y, size_y)
    
    #Distance between points
    dist = np.sqrt(dx * dx + dy * dy)

    #Fraction of distance moved, proportional to max_step (clamped to 1.0)
    move_frac = np.minimum(1.0, max_step / (dist + 1e-8))

    #Move and wrap around torus
    new_x = (pos_x + dx * move_frac) % size_x
    new_y = (pos_y + dy * move_frac) % size_y
    return new_x, new_y

