import sys
sys.path.append(r'..')
from Classes import Universe, Population, Cell
import numpy as np

u = Universe(100, 100)
u.gen_water_sources()
u.spawn_food(40)
cells = [Cell(np.random.uniform(0, 100), np.random.uniform(0, 100), energy=18, hydration=12) for _ in range(12)]
pop = Population(cells, u)
print('initial alive:', len(pop.cells))
for i in range(5):
    stats = pop.step()
    print(f'step {i+1}: alive={stats["alive"]} births_step={stats["births_step"]} deaths_step={stats["deaths_step"]} food_eaten_step={stats["food_eaten_step"]}')
