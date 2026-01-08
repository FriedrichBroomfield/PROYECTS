import numpy as np
import random
import pandas as pd
import os
import pathlib as path

from Params import *

# -------------- BASIC UTILITIES FOR CREATURES -------------- #

def ENERGY_stock(food_stock,water_stock,food_value=FOOD_ENERGY_VALUE,water_value=WATER_ENERGY_VALUE, food_relevance=FOOD_RELEVANCE,water_relevance=WATER_RELEVANCE):
    """Calculate energy stock from food amount and water amount."""
    energy = ((food_stock * food_value)**food_relevance) * ((water_stock * water_value)**water_relevance)
    return energy

def ENERGY_delta(food_stock,water_stock,food_delta=0,water_delta=0,food_relevance=FOOD_RELEVANCE,water_relevance=WATER_RELEVANCE):
    """Calculate energy gain from food amount and water amount."""
    delta_E = ENERGY_stock(food_stock,water_stock,food_relevance=food_relevance,water_relevance=water_relevance)*(food_relevance*food_delta/food_stock + water_relevance*water_delta/water_stock)
    return delta_E

def ENERGY_act(Energy,food_stock,water_stock,food_delta=0,water_delta=0):
    """Calculate new energy after food and water changes."""
    delta_E = ENERGY_delta(food_stock,water_stock,food_delta,water_delta)
    return Energy + delta_E


def torus_delta(coord1: float, 
                 coord2: float,
                    grid_size: float
                    ):
    """Calculate the toroidal delta between two coordinates.Univariate version of torus distance."""
    delta = coord1 - coord2
    if abs(delta) > grid_size / 2:
        if delta > 0:
            delta -= grid_size
        else:
            delta += grid_size
    return delta



def torus_distance(pos1: tuple, 
                     pos2: tuple,
                     grid_size_x=UNIVERSE_SIZE_X, 
                     grid_size_y=UNIVERSE_SIZE_Y
                     ):
    """Calculate the toroidal distance between two positions."""
    dx = torus_delta(pos1[0], pos2[0], grid_size_x)
    dy = torus_delta(pos1[1], pos2[1], grid_size_y)
    return np.sqrt(dx * dx + dy * dy)

def shortest_torus_vector(pos1: tuple, 
                           pos2: tuple,
                            grid_size_x=UNIVERSE_SIZE_X,
                            grid_size_y=UNIVERSE_SIZE_Y
                            ):
    """Calculate the shortest vector from pos1 to pos2 in a toroidal universe."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    if abs(dx) > grid_size_x / 2:
        if dx > 0:
            dx -= grid_size_x
        else:
            dx += grid_size_x
    if abs(dy) > grid_size_y / 2:
        if dy > 0:
            dy -= grid_size_y
        else:
            dy += grid_size_y
    return (dx, dy)

def colision_detection(pos1: tuple, 
                       pos2: tuple,
                       threshold=COLISION_THRESHOLD,
                       grid_size_x=UNIVERSE_SIZE_X, 
                       grid_size_y=UNIVERSE_SIZE_Y
                       ):
    """Detect if two positions are within a certain threshold."""
    distance = torus_distance(pos1, pos2, grid_size_x, grid_size_y)
    return distance < threshold

def move_towards(current_pos: tuple, 
                  target_pos: tuple,
                    step_size: float,
                    grid_size_x=UNIVERSE_SIZE_X,
                    grid_size_y=UNIVERSE_SIZE_Y
                    ):
    """Returns a Vector to the target position with a maximum length of step_size"""
    vector = shortest_torus_vector(current_pos, target_pos, grid_size_x, grid_size_y)

    # Shortes tor distance
    distance = np.sqrt(vector[0]**2 + vector[1]**2)

    if distance < 1e-9:
        return current_pos  # Already at the target

    vector = (vector[0] / distance * step_size, vector[1] / distance * step_size)
    return vector

def colision_response(pos1: tuple, 
                      pos2: tuple,
                        grid_size_x=UNIVERSE_SIZE_X,
                        grid_size_y=UNIVERSE_SIZE_Y
                      ):
    """Respond to a collision between two cells: No overlap allowed, between them they should move apart."""
    # Placeholder for collision response logic
    pass


# -------------- VISUALIZATION UTILITIES -------------- #

# SIZES OF THINGS FROM LARGEST TO SMALLEST: Water Source, Food, Creature
# OBJECTIVE: Sizes should scale with the size of the universe for better visualization in animations

def creature_radius(grid_size_x=UNIVERSE_SIZE_X, 
                     grid_size_y=UNIVERSE_SIZE_Y):
    """Calculate the radius of a creature based the size of the grid."""
    return (grid_size_x + grid_size_y) / 200

def food_radius(grid_size_x=UNIVERSE_SIZE_X, 
                     grid_size_y=UNIVERSE_SIZE_Y):
    """Calculate the radius of food based the size of the grid."""
    return (grid_size_x + grid_size_y) / 300

def water_radius(grid_size_x=UNIVERSE_SIZE_X, 
                     grid_size_y=UNIVERSE_SIZE_Y):
    """Calculate the radius of a water source based the size of the grid."""
    return (grid_size_x + grid_size_y) / 100






