import numpy as np
import matplotlib.pyplot as plt
import earth
import water
from mayavi import mlab


def out_of_bounds(world, rain):
    if rain.x_pos < 1 or rain.x_pos > world.lx - 2 or rain.y_pos < 1 or rain.y_pos > world.ly - 2:
        return True
    else:
        return False


iterations = 100
width, height = 1000, 1000
radius = 10

w = earth.World(width, height)
w.generate_height(scale=400.0, rand=False)
init_map = np.copy(w.height_map)
w.mayavi_plot(new_fig=True)

a_cloud = water.RainCloud(w, 20)
rain_drops = a_cloud.make_it_rain()
# mlab.figure(size=(800, 640))
for d, drop in enumerate(rain_drops):
    print(f'Simulating drop {d+1}')
    # xdata, ydata, zdata = [], [], []
    for i in range(iterations):
        if out_of_bounds(w, drop):
            break
        drop.roll()
        if out_of_bounds(w, drop):
            break
        drop.erode_radius(radius=radius)
    #     xdata.append(drop.x_pos)
    #     ydata.append(drop.y_pos)
    #     zdata.append(water.WaterDroplet.calc_height_and_grad(drop, w)[0]*40)
    # mlab.points3d(xdata, ydata, zdata, color=(0, 0.2, 1), scale_factor=.5)  # line_width=100.0, tube_radius=.25

final_map = w.height_map
w.mayavi_plot(new_fig=True)
mlab.show()

