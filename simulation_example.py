import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import earth
import water
# from mayavi import mlab

reload(earth)
reload(water)


def out_of_bounds(world, rain):
    if rain.pos[0] < 1 or rain.pos[0] > world.lx - 2 or rain.pos[1] < 1 or rain.pos[1] > world.ly - 2:
        return True
    else:
        return False


# @mlab.animate
def animate(n_iterations, water_droplets, terrain):
    for n, droplet in enumerate(water_droplets):
        print(f'Simulating drop {n+1}')
        iteration = 0
        while droplet.water > 0 and iteration < n_iterations:
            if out_of_bounds(terrain, droplet):
                break
            droplet.roll(terrain)
            if out_of_bounds(terrain, droplet):
                break
            droplet.erode_radius(terrain, radius=10.0)
            droplet.evapourate()

            iteration += 1
        print(iteration)
        # mlab_obj.mlab_source.scalars = terrain.height_map


# User Parameters
max_iterations = 500
n_drops = 50
width, height = 1000, 1000
radius = 3.0

w = earth.World(width, height)
w.generate_height(scale=400.0, rand=True)
init_map = np.copy(w.height_map)
# s = w.mayavi_plot(new_fig=True)
# w.show_map()

a_cloud = water.RainCloud(w, n_drops)
rain_drops = a_cloud.make_it_rain(strength=[0.1, 0.3])

animate(max_iterations, rain_drops, w)

final_map = w.height_map
print(f"The map has lost {np.sum(final_map - init_map)} material")
# w.mayavi_plot(new_fig=True)
# w.show_map()
# mlab.show()

plt.figure()
plt.imshow(final_map - init_map)
