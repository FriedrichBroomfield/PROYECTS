from __future__ import annotations
# Parameters.py
# -*- coding: utf-8 -*-
"""
Centralized simulation parameters.

This file is intentionally explicit: every parameter used by the simulation
is defined here (no hidden defaults).
"""

# ----------------------------
# World geometry
# ----------------------------
GRID_SIZE_X: int = 100
GRID_SIZE_Y: int = 100

NEST_RADIUS: float = 10.0  # Nest centered at (GRID_SIZE_X/2, GRID_SIZE_Y/2)

# Water sources
N_WATER_SOURCES: int = 4
WATER_SOURCES_RADIUS: float = 6.0  # for visualization + "blob" spawning exclusion
WATER_DRINK_RADIUS: float = 2.0    # must be > SOURCE_REPULSION_RADIUS

# Anti-crowding around water sources (repel near the center)
SOURCE_REPULSION_RADIUS: float = 0.7
THETA_SOURCE_REPULSION: float = 2.0  # strength

# ----------------------------
# Food
# ----------------------------
FOOD_VALUE: float = 12.0
FOOD_EAT_RADIUS: float = 1.5

MAX_FOOD: int = 250
FOOD_SPAWN_PER_STEP: int = 8

# Soft minimum separation between food particles (spawn only)
FOOD_MIN_SEP: float = 1.0

# ----------------------------
# Two types: A (slow, big vision) / B (fast, small vision)
# ----------------------------
TYPE_A: int = 0
TYPE_B: int = 1

TYPE_A_NAME: str = "A"
TYPE_B_NAME: str = "B"

# "Rapidez": distance moved per step
STEP_SIZE_A: float = 1.2
STEP_SIZE_B: float = 2.6

# Vision radius (perception radius)
VISION_RADIUS_A: float = 16.0
VISION_RADIUS_B: float = 10.0

# Heading inertia (higher = smoother turns, less jitter)
HEADING_ALPHA_A: float = 0.25
HEADING_ALPHA_B: float = 0.18

# Exploration noise strength when signals cancel (anti-paralysis)
OMEGA0_A: float = 0.25
OMEGA0_B: float = 0.45
G1_CANCEL_SCALE: float = 0.25

# Exploration persistence baseline (fast/low-vision should be more ballistic)
PERSIST_BASE_A: int = 8
PERSIST_BASE_B: int = 16
C_SEEK_PERSIST: float = 2.0  # multiplies (pi_w * L_w)

# ----------------------------
# Kernels / numerics
# ----------------------------
EPS: float = 1e-6
EPS_DIR: float = 1e-9

# Attraction kernel length scale is set per-agent: ell = vision/2
# psi(d) = exp(-d/ell)
# Repulsion kernel (agents, sources): eta(d) = exp(-(d/r)^2)

# ----------------------------
# Needs: urgency weights and satiety gating
# ----------------------------
URGENCY_P: float = 1.0  # U = 1/(resource+eps)^p

# Satiety threshold and slope (AND gate)
ENERGY_SAT: float = 35.0
HYDRATION_SAT: float = 35.0
SAT_SIGMOID_K: float = 0.35

# Water visibility gating
G0_WATER_SIGNAL: float = 0.35  # scale for L_w = g0/(||a_w_vis||+g0)

# Memory (water)
ETA_MEM_UPDATE: float = 0.60
DELTA_MEM_DECAY_A: float = 0.02
DELTA_MEM_DECAY_B: float = 0.03
KAPPA_MEM: float = 1.0
KAPPA_XI: float = 0.7

# ----------------------------
# Social / mate dynamics
# ----------------------------
INTERACTION_RADIUS: float = 1.5  # r_e for mate selection + eating overlap scale

AGENT_REPULSION_RADIUS: float = 1.0
THETA_AGENT_REPULSION_DEFAULT: float = 1.2

# Basal "inconsciente" attraction to same/different type
THETA_SAME_DEFAULT: float = 0.25*0
THETA_DIFF_DEFAULT: float = 0.15*0 

# Mate attraction strength (only active under joint satiety)
THETA_MATE_DEFAULT: float = 0.60
P_MATE_DEFAULT: float = 0.5  # preference for same-type mates (if same -> p; if diff -> 1-p)

# ----------------------------
# Metabolism / costs
# ----------------------------
# Costs are applied to BOTH resources (energy, hydration).
MOVE_COST_PER_DISTANCE: float = 0.18  # multiplied by step_size; subtracted from both

BASE_ENERGY_DECAY: float = 0.02
BASE_HYDRATION_DECAY: float = 0.01

LOW_HYDRATION_PENALTY_ENERGY: float = 0.02
LOW_HYDRATION_THRESHOLD: float = 3.0

ENERGY_MAX: float = 60.0
HYDRATION_MAX: float = 60.0

# Drinking gain per step if inside drink radius of any source
DRINK_GAIN: float = 3.5

# ----------------------------
# Reproduction (optional, partner-based)
# ----------------------------
ENABLE_REPRODUCTION: bool = True
REPRO_REQUIRE_NEST: bool = False  # if True, both parents must be in nest
REPRO_BETA: float = 0.02          # per-step probability scale
REPRO_COOLDOWN_STEPS: int = 220

REPRO_COST_ENERGY: float = 8.0
REPRO_COST_HYDRATION: float = 8.0

CHILD_SHARE: float = 0.45  # fraction transferred to child from each parent (symmetric)
CHILD_SPAWN_JITTER: float = 1.0  # spawn within [-1,1] box around parent

# ----------------------------
# Visualization
# ----------------------------
# Colores por tipo (fijos)
TYPE_A_COLOR: str = "#39C0FF"  # cyan
TYPE_B_COLOR: str = "#FF7A3D"  # orange

# Food color (fijo)
FOOD_COLOR: str = "#B7FF4A"

# Tamaños base
CELL_SIZE_BASE: float = 10.0
CELL_SIZE_ENERGY_SCALE: float = 2.0
CELL_SIZE_HYDR_SCALE: float = 1.2
GLOW_MULT: float = 3.0

HUD_FONT_SIZE: int = 10

# ----------------------------
# Hard collisions (no-overlap)
# ----------------------------
AGENT_COLLISION_RADIUS: float = 0.55   # radio físico por agente (ajusta a gusto)
COLLISION_ITERS: int = 5               # iteraciones del solver por step
COLLISION_EPS: float = 1e-9            # evita división por cero
COLLISION_DAMPING: float = 1.0         # 1.0 = corrección completa; <1 = más suave
