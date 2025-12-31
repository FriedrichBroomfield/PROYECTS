import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Universe:
    """Class representing the universe in the simulation:
    Caracteristics:
    * Toroidal 2D space where entities exist and interact.
    * Generates food resources at random positions.
    * Have water in fixed positions.
    * Has a fixed size and wraps around edges.
    * Comunal nest: circular in the center.
    * Entities can move, consume food, drink water, and reproduce within this space.
    * All positions are represented in Cartesian coordinates (x, y).
    
    Food:
    * Food items are generated at random positions within the universe.
    * Food items have a fixed energy value when consumed.
    * Food items are represented as points in the 2D space: Green circles.
    * Food gets replenished over time at random locations.
    * Food items have a limited lifetime before they disappear: 1 cicle.
    
    Water:
    * Water sources are located at fixed areas in the universe: Blue circles.
    * Water sources provide hydration to entities.
    * Water sources are represented as larger points in the 2D space.
    * Water is infite and does not get depleted.
    * Fixed area: Circular areas with a defined radius.
    
    Nest:
    * Nest provide a safe area for entities to reproduce.
    * Nest is represented as  a circle in the 2D space: White circle.
    * Nest is located at the center of the universe: origin (0,0)
    * Nest has a defined radius: 5% of the universe size.
    
    Cicle of the universe:
    * Every cicle, have a number of steps
    * Every step, entities can move, eat, drink, reproduce, or die.
    * 1 activity per step.
    * Every cicle have a defined number of steps: DAY_STEPS and NIGHT_STEPS: We asume NIGHT_STEPS=0, no activity at night.
    * DAY_STEPS = 1000 steps.
    * NIGHT_STEPS = 0 steps.
    * Both are configurable as parameters.
    
    """
    
    def __init__(self, size_x, size_y, food_value, nest_radius, water_sources_radius):
        self.size_x = size_x
        self.size_y = size_y
        self.food_value = food_value
        self.nest_radius = nest_radius
        self.water_sources_radius = water_sources_radius

        self.food_x = np.array([])
        self.food_y = np.array([])
        self.food_positions = list(zip(self.food_x, self.food_y))
    
    def gen_water_sources(self):
        """Generate fixed water sources at random positions.
        * 4 water sources at fixed positions.
        * 1 per quadrant.
        """
        margin = 0.1  # Margin from edges
        positions = [
            (margin * self.size_x, margin * self.size_y),
            ((1 - margin) * self.size_x, margin * self.size_y),
            (margin * self.size_x, (1 - margin) * self.size_y),
            ((1 - margin) * self.size_x, (1 - margin) * self.size_y),
        ]
        self.water_sources = np.array(positions)
    
    def gen_food(self, n_food):
        """Generate food items at random positions within the universe.
        * Food must be generated at random positions.
        * Food must be separated from water sources by at least water_sources_radius.
        * Food particles must be within the universe bounds.
        * Food particles must not overlap with each other by at least 1 unit.|
        """
        food_x = []
        food_y = []
        attempts = 0
        max_attempts = n_food * 10  # Prevent infinite loops

        while len(food_x) < n_food and attempts < max_attempts:
            x = np.random.uniform(0, self.size_x)
            y = np.random.uniform(0, self.size_y)

            # Check distance from water sources
            too_close_to_water = False
            for wx, wy in self.water_sources:
                dist = np.sqrt((x - wx) ** 2 + (y - wy) ** 2)
                if dist < self.water_sources_radius:
                    too_close_to_water = True
                    break

            if too_close_to_water:
                attempts += 1
                continue

            # Check distance from existing food
            too_close_to_food = False
            for fx, fy in zip(food_x, food_y):
                dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                if dist < 1.0:  # Minimum separation between food items
                    too_close_to_food = True
                    break

            if too_close_to_food:
                attempts += 1
                continue

            food_x.append(x)
            food_y.append(y)
            attempts += 1

        self.food_x = np.array(food_x)
        self.food_y = np.array(food_y)
        
# ===============================================================================
# ===============================================================================

class Cell:
    """Class representing a cell in the simulation.
    Caracteristics:
    * Each cell has a position (x, y) in the universe.
    * Each needs water and food to survive: energy level is: E = 0.7*ln(food)+0.3*ln(water)
    * Cells can move, consume food, drink water, and reproduce.
    * Cells lives under a dayle cycle of the universe: they can act freely, they decide to eat/drink/reproduce
    * Cells can die if their energy reaches a threshold.
    * Cells can reproduce if they have enough energy and are in the nest area.
    """
    
    def __init__(self, x, y, energy, death_energy_threshold=5):
        self.x = x
        self.y = y
        self.position = (x, y)
        self.energy = energy
        self.alive = True
        self.ready_to_repro = False
        self.needs_food = True
        self.needs_water = True
        self.death_energy_threshold = death_energy_threshold

    def brownian_move(self, step_size, universe):
        """Move the cell in a random direction by a step size.
        * The movement is toroidal: if the cell goes out of bounds, it wraps around.
        * cartesian coordinates (x, y).
        """
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)

        self.x = (self.x + dx) % universe.size_x
        self.y = (self.y + dy) % universe.size_y
        self.position = (self.x, self.y)
    
    def consume_food(self, universe: Universe):
        """Consume food if the cell is close enough to a food item.
        * If the cell is within a certain distance of a food item, it consumes it.
        * The cell gains energy equal to the food value of the universe.
        * The food item is removed from the universe.
        """
        consumption_distance = 1.0  # Distance within which the cell can consume food
        for i in range(len(universe.food_x)):
            fx,fy = universe.food_x[i], universe.food_y[i]
            dist = np.sqrt((self.x - fx) ** 2 + (self.y - fy) ** 2)
            if dist < consumption_distance:
                self.energy += universe.food_value
                # Remove the food item from the universe
                universe.food_x = np.delete(universe.food_x, i)
                universe.food_y = np.delete(universe.food_y, i)
                return True  # Food consumed
        return False  # No food consumed
    
    def drink_water(self, universe: Universe):
        """Drink water if the cell is close enough to a water source.
        * If the cell is within a certain distance of a water source, it drinks.
        * The cell gains hydration (not modeled here).
        """
        drinking_distance = 2.0  # Distance within which the cell can drink water
        for wx, wy in universe.water_sources:
            dist = np.sqrt((self.x - wx) ** 2 + (self.y - wy) ** 2)
            if dist < drinking_distance:
                # Cell drinks water (hydration not modeled)
                return True  # Water drunk
        return False  # No water drunk
    
    def check_reproduction(self, universe: Universe, repro_energy_threshold):
        """Check if the cell can reproduce.
        * The cell can reproduce if it has enough energy and is within the nest area.
        """
        dist_to_nest = np.sqrt((self.x - universe.size_x / 2) ** 2 + (self.y - universe.size_y / 2) ** 2)
        if self.energy >= repro_energy_threshold and dist_to_nest <= universe.nest_radius:
            self.ready_to_repro = True
        else:
            self.ready_to_repro = False
            
    def reproduce(self):
        """Reproduce a new cell.
        * The new cell is created at the same position with half the energy of the parent.
        * The parent loses half its energy.
        """
        random_offset = np.random.uniform(-1, 1, size=2)
        if self.ready_to_repro:
            offspring_energy = self.energy / 2
            self.energy /= 2
            return Cell(self.x + random_offset[0], self.y + random_offset[1], offspring_energy)
        return None
    
    def check_death(self):
        """Check if the cell dies.
        * The cell dies if its energy falls below the death threshold.
        """
        if self.energy < self.death_energy_threshold:
            self.alive = False
            return True
        return False
    
class Population:
    """Class representing a population of cells in the simulation.
    Caracteristics:
    * Manages a collection of Cell instances.
    * Agregates statistics about the population.
    * Handles cell-level operations such as movement, eating, drinking, reproduction, and death.
    * Handles population-level operations such as adding new cells and removing dead cells.
    * Provides methods to get population statistics.
    * Supports different cell types with distinct behaviors and characteristics.
    * Calls the Universe methods to interact with the environment.
    * Calls the Cells methods.
    * Provides methods to states of the population at every step.
    """
    def __init__(self, cells: list, universe: Universe):
        self.cells = cells
        self.cells_alive_now = len(cells)
        self.cells_dead_now = 0
        
        self.death_causes = {}
        self.total_births = 0
        self.total_deaths = 0
        
        self.ready_to_repro_count = 0
        
        self.food_eaten_today = 0
        self.universe = universe
        
        self.population_percentage_changed = 0.0
        
    def size(self):
        return len(self.cells)
    
    def populate_stats(self):
        """Get statistics about the population.
        * Total number of cells.
        * Number of alive cells.
        * Number of dead cells.
        * Number of cells ready to reproduce.
        * Average energy level.
        
        * Growth rate: percentage change in population size since last step.
        * Death rate: percentage of cells that died since last step.
        * Birth rate: percentage of new cells born since last step.
        """
        total_cells = len(self.cells)
        alive_cells = sum(1 for cell in self.cells if cell.alive)
        dead_cells = total_cells - alive_cells
        ready_to_repro = sum(1 for cell in self.cells if cell.ready_to_repro)
        avg_energy = np.mean([cell.energy for cell in self.cells if cell.alive]) if alive_cells > 0 else 0.0
        
        growth_rate = 100.0 * (alive_cells - self.cells_alive_now) / self.cells_alive_now if self.cells_alive_now > 0 else 0.0
        death_rate = 100.0 * (dead_cells - self.cells_dead_now) / self.cells_alive_now if self.cells_alive_now > 0 else 0.0
        birth_rate = 100.0 * (total_cells - self.total_births) / self.cells_alive_now if self.cells_alive_now > 0 else 0.0
        
        self.population_percentage_changed = growth_rate
        
        self.cells_alive_now = alive_cells
        self.cells_dead_now = dead_cells
        self.total_births = total_cells
        self.total_deaths += dead_cells
        
        stats = {
            "total_cells": total_cells,
            "alive_cells": alive_cells,
            "dead_cells": dead_cells,
            "ready_to_repro": ready_to_repro,
            "avg_energy": avg_energy,
            "growth_rate": growth_rate,
            "death_rate": death_rate,
            "birth_rate": birth_rate
        }
        
        return stats
    
    def get_positions_energies(self):
        """Get the positions and energies of all alive cells.
        * Returns arrays of x positions, y positions, and energies.
        """
        x_positions = np.array([cell.x for cell in self.cells if cell.alive])
        y_positions = np.array([cell.y for cell in self.cells if cell.alive])
        energies = np.array([cell.energy for cell in self.cells if cell.alive])
        
        return x_positions, y_positions, energies
    
    def step(self, repro_energy_threshold, step_size):
        """Perform a simulation step for the population.
        * Each cell moves, eats, drinks, checks reproduction, and checks death.
        * 1 activity per step.
        * New cells from reproduction are added to the population.
        * Dead cells are removed from the population.
        * Updates statistics about the population.
        * Updates the simulation state.
        """
        new_cells = []
        for cell in self.cells:
            if not cell.alive:
                continue
            
            # Move
            cell.brownian_move(step_size, self.universe)
            
            # Eat
            if cell.needs_food:
                ate = cell.consume_food(self.universe)
                if ate:
                    self.food_eaten_today += 1
            
            # Drink
            if cell.needs_water:
                cell.drink_water(self.universe)
            
            # Check reproduction
            cell.check_reproduction(self.universe, repro_energy_threshold)
            if cell.ready_to_repro:
                offspring = cell.reproduce()
                if offspring:
                    new_cells.append(offspring)
            
            # Check death
            cell.check_death()
        
        # Remove dead cells
        self.cells = [cell for cell in self.cells if cell.alive]
        
        # Add new cells from reproduction
        self.cells.extend(new_cells)
        
        # Update statistics
        stats = self.populate_stats()
        stats['food_eaten_today'] = self.food_eaten_today
        
        return stats
    
# ===============================================================================
# ===============================================================================
class Simulation:
    """Class representing the overall simulation.
    Characteristics:
    * Manages the universe and population.
    * Runs the simulation for a defined number of cycles and steps.
    * Collects and stores statistics at each step.
    * Provides methods to visualize the simulation state.
    * Simulation shows every step of the simulation using matplotlib.
    """
    
    def __init__(self, universe: Universe, population: Population, day_steps=1000, night_steps=0):
        self.universe = universe
        self.population = population
        self.day_steps = day_steps
        self.night_steps = night_steps
        self.stats_history = []
    
    def run_cycle(self, repro_energy_threshold, step_size):
        """Run a full cycle of the simulation (day + night).
        * During the day, cells can act (move, eat, drink, reproduce, die).
        * During the night, no activity occurs.
        * Collect statistics at each step.
        """
        # Daytime steps
        for _ in range(self.day_steps):
            stats = self.population.step(repro_energy_threshold, step_size)
            # record positions for animation
            fx = self.universe.food_x.copy() if getattr(self.universe, 'food_x', None) is not None else np.array([])
            fy = self.universe.food_y.copy() if getattr(self.universe, 'food_y', None) is not None else np.array([])
            cx, cy, _ = self.population.get_positions_energies()
            stats['food_x'] = fx.tolist()
            stats['food_y'] = fy.tolist()
            stats['cell_x'] = cx.tolist()
            stats['cell_y'] = cy.tolist()
            self.stats_history.append(stats)
        
        # Nighttime steps (no activity)
        for _ in range(self.night_steps):
            stats = self.population.populate_stats()
            fx = self.universe.food_x.copy() if getattr(self.universe, 'food_x', None) is not None else np.array([])
            fy = self.universe.food_y.copy() if getattr(self.universe, 'food_y', None) is not None else np.array([])
            cx, cy, _ = self.population.get_positions_energies()
            stats['food_x'] = fx.tolist()
            stats['food_y'] = fy.tolist()
            stats['cell_x'] = cx.tolist()
            stats['cell_y'] = cy.tolist()
            self.stats_history.append(stats)
    
    def get_stats_dataframe(self):
        """Get the collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_history)
    
    def visualize(self):
        """Visualize the simulation state using matplotlib.
        * Shows the universe with food, water sources, nest, and cells.
        * Animates the simulation over time using recorded positions in stats_history.
        """
        if not self.stats_history:
            print("No recorded steps to visualize.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.universe.size_x)
        ax.set_ylim(0, self.universe.size_y)
        ax.set_title("Evolutionary Simulation")
        plt.style.use("dark_background")
        
        # Plot water sources
        for wx, wy in self.universe.water_sources:
            water_circle = plt.Circle((wx, wy), self.universe.water_sources_radius, color='blue', alpha=0.5)
            ax.add_artist(water_circle)
        
        # Plot nest
        nest_circle = plt.Circle((self.universe.size_x / 2, self.universe.size_y / 2), self.universe.nest_radius, color='white', alpha=0.5)
        ax.add_artist(nest_circle)
        
        food_scatter = ax.scatter([], [], c='green', s=50, label='Food')
        cell_scatter = ax.scatter([], [], c='red', s=20, label='Cells')
        
        def init():
            empty = np.empty((0, 2))
            food_scatter.set_offsets(empty)
            cell_scatter.set_offsets(empty)
            return food_scatter, cell_scatter
        
        def update(frame):
            entry = self.stats_history[frame]
            fx = np.array(entry.get('food_x', []))
            fy = np.array(entry.get('food_y', []))
            if fx.size and fy.size:
                food_positions = np.column_stack((fx, fy))
            else:
                food_positions = np.empty((0, 2))
            food_scatter.set_offsets(food_positions)

            cx = np.array(entry.get('cell_x', []))
            cy = np.array(entry.get('cell_y', []))
            if cx.size and cy.size:
                cell_positions = np.column_stack((cx, cy))
            else:
                cell_positions = np.empty((0, 2))
            cell_scatter.set_offsets(cell_positions)

            return food_scatter, cell_scatter
        
        self.ani = FuncAnimation(fig, update, frames=range(len(self.stats_history)), init_func=init, blit=True, interval=100)
        plt.legend()
        plt.show()
        
# ===============================================================================
# ===============================================================================
# Example usage:


if __name__ == "__main__":
    # Create universe
    universe = Universe(size_x=100, size_y=100, food_value=10, nest_radius=5, water_sources_radius=10)
    universe.gen_water_sources()
    universe.gen_food(n_food=50)
    
    # Create initial population
    initial_cells = [Cell(x=np.random.uniform(0, 100), y=np.random.uniform(0, 100), energy=20) for _ in range(20)]
    population = Population(cells=initial_cells, universe=universe)
    
    # Create simulation
    simulation = Simulation(universe=universe, population=population, day_steps=100, night_steps=0)

    # Run simulation for 500 cycles
    for cycle in range(500):
        simulation.run_cycle(repro_energy_threshold=30, step_size=1.0)
    
    # Get statistics
    stats_df = simulation.get_stats_dataframe()
    print(pd.DataFrame(stats_df))
    
    # Visualize simulation
    simulation.visualize()