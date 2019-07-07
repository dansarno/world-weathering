import numpy as np
import matplotlib.pyplot as plt
import earth
import water
from mayavi import mlab

iterations = 500
width, height = 1000, 1000

w = earth.World(width, height)
w.generate_height(scale=400.0, rand=False)
init_map = np.copy(w.height_map)
w.mayavi_plot(new_fig=True)

a_cloud = water.RainCloud(w, 1000)
# TODO find bug where raindrop goes out of bounds
rain_drops = a_cloud.make_it_rain()
# test_drop = water.WaterDroplet(w)
# mlab.figure(size=(800, 640))
for d, drop in enumerate(rain_drops):
    print(f'Simulating drop {d+1}')
    # xdata, ydata, zdata = [], [], []
    # d_h = []
    for i in range(iterations):
        if drop.x_pos < 1 or drop.x_pos > w.lx - 2 or drop.y_pos < 1 or drop.y_pos > w.ly - 2:
            break
        drop.roll(w)
        if drop.x_pos < 1 or drop.x_pos > w.lx - 2 or drop.y_pos < 1 or drop.y_pos > w.ly - 2:
            break
        drop.erode2(w, radius=10.0)
        # xdata.append(drop.x_pos)
        # ydata.append(drop.y_pos)
        # zdata.append(water.WaterDroplet.calc_height_and_grad(drop, w)[0]*40)
    # mlab.points3d(xdata, ydata, zdata, color=(0, 0.2, 1), scale_factor=.5)  # line_width=100.0, tube_radius=.25

final_map = w.height_map
w.mayavi_plot(new_fig=True)
mlab.show()
