import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import seaborn as sns
from matplotlib.animation import FuncAnimation


random.seed(23032001)
np.random.seed(23032001)

# Grid (polar)
r_min, r_max = 0, 10
nr, ntheta = 200, 360
r = np.linspace(r_min, r_max, nr)
theta = np.linspace(0, 2 * np.pi, ntheta)
Theta, R = np.meshgrid(theta, r)

# Random points concentrated IN the outer ring, evenly distributed in angle
rng = np.random.default_rng(23032001)
N = 500
r_max = 100
r_points = np.full(N, r_max)
theta_points = rng.uniform(0.0, 2 * np.pi, N)
points = np.column_stack((r_points, theta_points))
print(points)



# Plot: Black background with white outer ring limit, blue points (static)
fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='polar')
ax.patch.set_facecolor('black')

# Blue points
ax.scatter(points[:, 1], points[:, 0], c='dodgerblue', s=40, edgecolors='white', linewidths=0.6, zorder=4)

# White outer ring
theta_full = np.linspace(0.0, 2 * np.pi, 400)
ax.plot(theta_full, np.full_like(theta_full, r_max), color='white', linewidth=1.6, zorder=5)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.grid(False)

plt.title('Polar grid with outer-ring points', color='white')
plt.show()

# Animation: use a separate polar axes and disable blitting (blit can be problematic with polar)
num_steps = 100
step_radius = r_max / num_steps

# New figure/axes for animation to avoid reusing the displayed static axes
fig2 = plt.figure(figsize=(8, 8))
fig2.patch.set_facecolor('black')
ax2 = fig2.add_subplot(111, projection='polar')
ax2.patch.set_facecolor('black')

ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.grid(False)
ax2.plot(theta_full, np.full_like(theta_full, r_max), color='white', linewidth=1.6, zorder=5)

r_points_init = r_points.copy()
sc = ax2.scatter(theta_points, r_points_init, c='dodgerblue', s=40, edgecolors='white', linewidths=0.6, zorder=4)

def update(frame):
    r_frame = np.maximum(0, r_points_init - step_radius * frame)
    sc.set_offsets(np.column_stack((theta_points, r_frame)))
    return (sc,)

anim = FuncAnimation(fig2, update, frames=num_steps, interval=10, blit=False)
plt.title('Points moving inward', color='white')
plt.show()

