# Parameters.py
# -*- coding: utf-8 -*-

from __future__ import annotations

# ----------------------------
# World geometry
# ----------------------------
GRID_SIZE_X: int = 100
GRID_SIZE_Y: int = 100

# Water sources
N_WATER_SOURCES: int = 4
WATER_SOURCES_RADIUS: float = 6.0   # solo visual / exclusión spawn food
WATER_DRINK_RADIUS: float = 2.2     # radio para "beber" (centro de la fuente)

# ----------------------------
# Food
# ----------------------------
MAX_FOOD: int = 250
FOOD_SPAWN_PER_STEP: int = 8
FOOD_MIN_SEP: float = 1.0
FOOD_EAT_RADIUS: float = 1.2

EAT_GAIN_F: float = 10.0   # +f (stock comida) por comer 1 partícula
DRINK_GAIN_W: float = 3.2  # +w (stock agua) por tick si estás dentro de radio de beber

# ----------------------------
# Two types: A (slow, big vision) / B (fast, small vision)
# ----------------------------
TYPE_A: int = 0
TYPE_B: int = 1

STEP_SIZE_A: float = 1.2
STEP_SIZE_B: float = 2.0 # antes 2.6

VISION_RADIUS_A: float = 16.0
VISION_RADIUS_B: float = 10.0 # antes 8.0

# Heading inertia (suavizado de dirección)
HEADING_ALPHA_A: float = 0.25
HEADING_ALPHA_B: float = 0.25 #antes 0.18

# Ruido exploratorio
OMEGA0_A: float = 0.25
OMEGA0_B: float = 0.45
G1_CANCEL_SCALE: float = 0.25

# Persistencia exploratoria (búsqueda balística)
PERSIST_BASE_A: int = 8
PERSIST_BASE_B: int = 16
C_SEEK_PERSIST: float = 2.0

# ----------------------------
# Numerics
# ----------------------------
EPS: float = 1e-6
EPS_DIR: float = 1e-9

# ----------------------------
# Stocks caps
# ----------------------------
W_MAX: float = 60.0
F_MAX: float = 60.0

# ----------------------------
# Energy aggregator E(w,f): CES or Cobb-Douglas
# ----------------------------
USE_CES: bool = True

# CES: E = (a w^rho + (1-a) f^rho)^(1/rho)
CES_alpha: float = 0.5
CES_rho: float = 0.25

# Cobb-Douglas: E = A w^a f^b
CD_A: float = 1.0
CD_a: float = 0.5
CD_b: float = 0.5

# ----------------------------
# Satiety gate (AND) en stocks, no en E/hidratación separadas
# ----------------------------
W_SAT: float = 35.0
F_SAT: float = 35.0
SAT_SIGMOID_K: float = 0.35

# ----------------------------
# Urgency weights (puedes usar marginales de E o stocks)
# ----------------------------
URGENCY_USE_MARGINALS: bool = True
URGENCY_P: float = 1.0  # si se usa por stocks: 1/(stock^p)

# ----------------------------
# Memory/search for water (cuando no hay agua visible)
# ----------------------------
G0_WATER_SIGNAL: float = 0.35
ETA_MEM_UPDATE: float = 0.60
DELTA_MEM_DECAY_A: float = 0.02
DELTA_MEM_DECAY_B: float = 0.03
KAPPA_MEM: float = 1.0
KAPPA_XI: float = 0.7

# ----------------------------
# Social / mate dynamics
# ----------------------------
INTERACTION_RADIUS: float = 1.5

# Solo para NO-overlap (colisión dura)
AGENT_RADIUS: float = 0.55
COLLISION_DIST: float = 2.0 * AGENT_RADIUS
COLLISION_GRID_CELL: float = COLLISION_DIST  # tamaño de celda grilla
COLLISION_ITERS: int = 2

# Distancia donde actúan interacciones "theta" (si quieres apagar debajo)
D_THETA: float = 6.0

# Repulsión corta para evitar "pegotes" (no es colisión)
AGENT_REPULSION_RADIUS: float = 1.2
THETA_AGENT_REPULSION: float = 0.8

# Thetas evolucionables (inicialmente 0)
THETA_SAME_INIT: float = 0.0
THETA_DIFF_INIT: float = 0.0

# Mutación (solo theta_same y theta_diff)
THETA_MUT_SIGMA: float = 0.5   # antes 0.02
THETA_MIN: float = -1.5  # permite negativos (repulsión)
THETA_MAX: float = 1.5

# Mate (opcional): activo solo con saciedad AND
ENABLE_REPRODUCTION: bool = True
REPRO_BETA: float = 1   # antes 0.02
REPRO_COOLDOWN_STEPS: int = 220

# Costos por reproducción (sobre stocks)
REPRO_COST_W: float = 6.0
REPRO_COST_F: float = 6.0

# Umbral mínimo para intentar reproducirse (además de satiety)
REPRO_W_MIN: float = 20.0
REPRO_F_MIN: float = 20.0

CHILD_SPAWN_JITTER: float = 1.0

# ----------------------------
# Metabolism / costs (costo por PASO, no distancia)
# ----------------------------
MOVE_COST_PER_STEP: float = 0.25
BASE_W_DECAY: float = 0.01
BASE_F_DECAY: float = 0.02

LOW_W_THRESHOLD: float = 3.0
LOW_W_PENALTY_F: float = 0.02

# ----------------------------
# Visualization
# ----------------------------
FIG_W: float = 16.0
FIG_H: float = 9.0
