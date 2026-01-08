""""
This module contains the parameters for the evolutionary simulation.
"""

# UTILITY params

epsilon = 1e-6  # Small value to avoid division by zero
psi = 1e-3      # Small value for numerical stability
xi = 0.1        # Small value for convergence criteria





# Environment settings for the simulation
NUM_CYCLES = 1000  # Total number of cycles in the simulation
STEPS_PER_CYCLE = 20  # Number of steps per cycle


UNIVERSE_SIZE_X = 900  # Size of the universe in the X dimension
UNIVERSE_SIZE_Y = 1600  # Size of the universe in the Y dimension

FOOD_MAX_AMOUNT = 500  # Maximum amount of food in the universe
FOOD_SPAWN_RATE = 5    # Rate at which food spawns per time unit
FOOD_ENERGY_VALUE = 15.0  # Energy value per unit of food

WATER_SOURCES = 50  # Number of water sources in the universe: Infinite water per source
WATER_RADIUS = 20  # Radius of each water source
WATER_ENERGY_VALUE = 8.0  # Energy value per unit of water



# Cost of being alive
COST_PER_STEP = 0.1  # Energy cost of moving one step
MOVES_PER_STEP = 1.0  # Movements per step

# Cell characteristics
DISTANCE_PER_MOVE  =  0  # Caracteristics of cell parameters: Some are faster than others
COLISION_THRESHOLD = 5.0  # Distance threshold for collision detection
INITIAL_ENERGY = 100.0  # Initial energy of each cell
WATER_RELEVANCE = 0.5  # Relevance of water consumption to overall energy
FOOD_RELEVANCE = 0.5  # Relevance of food consumption to overall energy

# IGORE PARAMS
SIGTH_RADIUS_IGOR = 50  # Sight radius for cells
VELOCITY_IGOR = 5  # Velocity of cells

# TIGGER PARAMS
SIGTH_RADIUS_TIGGER = 10  # Sight radius for cells
VELOCITY_TIGGER = 15  # Velocity of cells