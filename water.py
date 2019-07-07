import numpy as np
from random import random
import scipy.misc
import matplotlib.pyplot as plt
import earth
from numba import jit


class WaterDroplet:
    """
    Class docstring
    """
    INERTIA = 0.05  # At 0 water instantly changes direction. At 1, water will never change direction.

    def __init__(self, world, x_dir=0.0, y_dir=0.0, water=0.2, material=0.0, sediment_capacity=4.0):
        """

        Args:
            world:
        """
        self.world = world
        self.x_pos = random() * self.world.lx
        self.y_pos = random() * self.world.ly
        self.z_pos = 100.0  # TODO change this to something sensible i.e. initially high above the world
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.water = water
        self.material = material
        self.sediment_capacity = sediment_capacity
        self.d_h = 0

    def __repr__(self):
        """Returns representation of the object e.g. WaterDroplet(43.535, 28.114)"""
        return f'{self.__class__.__name__}({self.x_pos}, {self.y_pos})'

    def __str__(self):
        """Returns a human-readable string describing the droplet object"""
        return f'The water droplet is at: ({self.x_pos}, {self.y_pos})'

    def __add__(self, other):
        # TODO find a way of combining the two droplet positions
        d = self.water + other.water  # combine the water volume
        m = self.material + other.material  # combine the droplet's sediment
        x = self.x_dir + other.x_dir  # add the x components of the direction vectors
        y = self.y_dir + other.y_dir  # add the y components of the direction vectors
        # TODO remove the two instances of the droplets to be combined
        return WaterDroplet(self.world, x_dir=x, y_dir=y, water=d, material=m)

    # @jit(nopython=True)
    def roll(self, world):
        """
        Determines the drop's current height and 2d gradient and calculates the new drop position
        Returns:
        """
        init_height, x_grad, y_grad = WaterDroplet.calc_height_and_grad(self, world)
        # Update the droplet's direction and position (move position 1 unit regardless of speed)
        self.x_dir = (self.x_dir * self.INERTIA) - (x_grad * (1 - self.INERTIA))
        self.y_dir = (self.y_dir * self.INERTIA) - (y_grad * (1 - self.INERTIA))
        # Normalise direction
        mag = np.sqrt((self.x_dir ** 2) + (self.y_dir ** 2))
        if mag != 0:
            self.x_dir /= mag
            self.y_dir /= mag

        self.x_pos += self.x_dir
        self.y_pos += self.y_dir

        if self.x_pos < 1 or self.x_pos > self.world.lx - 2 or self.y_pos < 1 or self.y_pos > self.world.ly - 2:
            new_height = init_height
        else:
            new_height, x_grad, y_grad = WaterDroplet.calc_height_and_grad(self, world)

        # Update the d_h attribute of the water droplet
        self.d_h = init_height - new_height

    # @jit(nopython=True, parallel=True)
    def erode(self, world):
        init_height, x_grad, y_grad = WaterDroplet.calc_height_and_grad(self, world)
        pos1, pos2, pos3, pos4 = WaterDroplet._find_nodes_and_offsets(self.x_pos, self.y_pos)[0:4]
        drop_pos = (self.x_pos, self.y_pos)
        coords = [pos1, pos2, pos3, pos4]
        for coord in coords:
            dist = WaterDroplet._dist(drop_pos, coord)
            if not self.d_h < 0:
                world.height_map[coord[0]][coord[1]] -= self.d_h * (dist / np.sqrt(2)) * self.water

    def erode2(self, world, radius=3.0):
        drop_pos = (self.x_pos, self.y_pos)
        if not self.d_h < 0:
            weightings_dict = WaterDroplet._get_nodes_and_weights_in_raduis(world.height_map, drop_pos, radius)
            for pos, weight in weightings_dict.items():
                i, j = pos
                world.height_map[i][j] -= self.d_h * weight * self.water

    def evapourate(self):
        """Docstring"""
        pass

    def remove_drop(self):
        """Docstring"""
        pass

    @classmethod
    def change_inertia(cls, new_value):
        cls.INERTIA = new_value

    # @jit(nopython=True)
    def calc_height_and_grad(self, world):
        """
        Given the world in which the drop exists and the current (x,y) position, this static method calculates the
        current height of the drop and the 2d gradient. This is required to determine the gravitational force on the
        drop when establishing its next position.
        """
        world = world.height_map
        nw, ne, se, sw, x_offset, y_offset = WaterDroplet._find_nodes_and_offsets(self.x_pos, self.y_pos)
        height_nw = world[nw[0]][nw[1]]
        height_ne = world[ne[0]][ne[1]]
        height_sw = world[sw[0]][sw[1]]
        height_se = world[se[0]][se[1]]
        # Calculate droplet's direction of flow with bilinear interpolation of height difference along the edges
        x_grad = ((height_ne - height_nw) * (1 - y_offset)) + ((height_se - height_sw) * y_offset)
        y_grad = ((height_sw - height_nw) * (1 - x_offset)) + ((height_se - height_ne) * x_offset)
        # Calculate height with bilinear interpolation of the heights of the nodes of the cell
        self.z_pos = (height_nw * (1 - x_offset) * (1 - y_offset)) + \
                     (height_ne * x_offset * (1 - y_offset)) + \
                     (height_sw * (1 - x_offset) * y_offset) + \
                     (height_se * x_offset * y_offset)

        return self.z_pos, x_grad, y_grad

    @staticmethod
    @jit(nopython=True)
    def _find_nodes_and_offsets(x_pos, y_pos):
        x_coord = int(x_pos)
        y_coord = int(y_pos)
        # Calculate droplet's offset inside the cell (0,0) = at NW node, (1,1) = at SE node
        x_offset = x_pos - x_coord
        y_offset = y_pos - y_coord
        # Calculate heights of the four nodes of the droplet's cell
        x1 = x_coord
        x2 = x_coord + 1
        y1 = y_coord
        y2 = y_coord + 1
        nw = (x1, y1)
        ne = (x2, y1)
        se = (x2, y2)
        sw = (x1, y2)
        return nw, ne, se, sw, x_offset, y_offset
        # return x1, y1, x2, y2, x_offset, y_offset

    @staticmethod
    def _get_nodes_and_weights_in_raduis(heightmap, drop_position, radius, funct='gauss'):
        x, y = drop_position
        # Define subsection of map to search (+radius to -radius square)
        x_minus = np.ceil(x - radius) if np.ceil(x - radius) >= 0 else 0
        x_plus = np.floor(x + radius) if np.floor(x + radius) <= np.shape(heightmap)[0] - 1 else np.shape(heightmap)[0]
        y_minus = np.ceil(y - radius) if np.ceil(y - radius) >= 0 else 0
        y_plus = np.floor(y + radius) if np.floor(y + radius) <= np.shape(heightmap)[1] - 1 else np.shape(heightmap)[1]
        dict_position_and_weights = {}
        for i in range(int(x_minus), int(x_plus)):
            for j in range(int(y_minus), int(y_plus)):
                grid_point = (i, j)
                drop_to_gridpoint = WaterDroplet._dist(grid_point, drop_position)
                if drop_to_gridpoint < radius:
                    if funct == 'gauss':
                        weighting = WaterDroplet._gauss(drop_to_gridpoint, radius)
                        dict_position_and_weights[grid_point] = weighting
        return dict_position_and_weights

    @staticmethod
    @jit(nopython=True)
    def _dist(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    @staticmethod
    @jit(nopython=True)
    def _gauss(distance, radius):
        return np.exp(-np.power(distance, 2) / (2 * np.power(np.sqrt(radius) / 2, 2)))


class RainCloud(WaterDroplet):
    def __init__(self, world, num_droplets):
        """
        Args:
            world:
            num_droplets:
        """
        self.world = world
        self.strength = 0  # placeholder parameter
        self.num_droplets = num_droplets
        self.cloud = []

    def __repr__(self):
        """Returns representation of the object"""
        return f'{self.__class__.__name__}({self.world.__repr__()}, {self.num_droplets})'

    def __str__(self):
        """Returns a human-readable string describing the cloud object"""
        return f'This cloud, with {len(self.cloud)} droplets,' \
               f' has been generated in a world which is {self.world.lx} x {self.world.ly} pixels'

    def __len__(self):
        """Returns the number of rain drops in the cloud object"""
        return len(self.cloud)

    def __getitem__(self, item):
        """Returns the WaterDroplet object at that cloud element number"""
        return self.cloud[item]

    def make_it_rain(self):
        """ Creates an array of WaterDroplet objects and stores them in self.cloud list"""
        for _ in range(self.num_droplets):
            self.cloud.append(WaterDroplet(self.world))
        return self.cloud

    # def erode(self):
    #     for drop in self.cloud:
    #         super().erode()
    #     pass

    def print_droplets(self):
        """Prints the position of each droplet in the cloud"""
        for drop in self.cloud:
            print(drop)

    def show_cloud(self):
        """Displays an image of the cloud"""
        print(f'This cloud has {self.num_droplets} droplets')
        cloud_image = np.zeros((self.world.lx, self.world.ly))
        for drop in self.cloud:
            cloud_image[drop.x_pos, drop.y_pos] = 1.0
        scipy.misc.toimage(cloud_image).show()


# To be run in the for testing and debugging purposes
if __name__ == '__main__':
    width, height = 100, 100
    w = earth.World(width, height)
    w.generate_height()

    test_dict = WaterDroplet._get_nodes_and_weights_in_raduis(w.height_map, (45.2, 70.3), 6)
    print(test_dict)

    canvas = np.zeros(np.shape(w.height_map))
    for pos, weight in test_dict.items():
        print(pos, weight)
        i, j = pos
        canvas[i][j] = weight

    plt.imshow(canvas, cmap='bone')
    plt.show()

    # y = []
    # x_axis = np.arange(-5.0, 5.0, 0.1)
    # for x in x_axis:
    #     y.append(WaterDroplet._gauss(x, 4))
    #
    # plt.figure()
    # plt.plot(x_axis, y)
    # plt.show()

    # test_cloud = RainCloud(w, 100)
    # test_cloud.make_it_rain()

    # x = 0.5
    # y = 0.5
    # d = WaterDroplet(w, x, y)
    # pos1, pos2, pos3, pos4 = d._find_nodes_and_offsets(x, y)[:4]
    # coord = (x, y)
    # print(d._dist(pos1, coord))
