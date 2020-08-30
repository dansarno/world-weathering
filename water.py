import numpy as np
import random
import scipy.misc
import matplotlib.pyplot as plt
import earth


class WaterDroplet:
    """
    Class docstring
    """
    INERTIA = 0.05  # At 0 water instantly changes direction. At 1, water will never change direction.
    EVAP_RATE = 0.001

    def __init__(self, world, pos, vector=np.array([0.0, 0.0]), water=0.1):
        """

        Args:
            world:
        """
        self.world = world
        self.pos = pos
        self.z_pos = 100.0  # TODO change this to something sensible i.e. initially high above the world
        self.vector = vector
        self.water = water
        self.d_h = 0

    def __repr__(self):
        """Returns representation of the object e.g. WaterDroplet(43.535, 28.114)"""
        return f'{self.__class__.__name__}({self.pos[0]}, {self.pos[1]})'

    def __str__(self):
        """Returns a human-readable string describing the droplet object"""
        return f'The water droplet is at: ({self.pos[0]}, {self.pos[1]})'

    def __add__(self, other):
        new_water = self.water + other.water  # combine the water volume
        new_vect = self.vector + other.vector  # combine the water vectors
        # TODO remove the two instances of the droplets to be combined
        return WaterDroplet(self.world, self.pos, vector=new_vect, water=new_water)

    def roll(self):
        """
        Determines the drop's current height and 2d gradient and calculates the new drop position
        Returns:
        """
        init_height, grad_vector = WaterDroplet._calc_height_and_grad(self)
        # Update the droplet's direction and position (move position 1 unit regardless of speed)
        self.vector = (self.vector * self.INERTIA) - (grad_vector * (1 - self.INERTIA))

        # Normalise direction
        # direction_mag = np.linalg.norm(self.vector)
        direction_mag = np.sqrt(np.sum(self.vector ** 2))
        if direction_mag:
            self.vector /= direction_mag

        # Update position
        self.pos += self.vector

        if self.pos[0] < 1 or self.pos[0] > self.world.lx - 2 or self.pos[1] < 1 or self.pos[1] > self.world.ly - 2:
            new_height = init_height
        else:
            new_height = WaterDroplet._calc_height_and_grad(self)[0]

        # Update the d_h attribute of the water droplet
        self.d_h = init_height - new_height

    def erode(self):
        # init_height, grad_vector = WaterDroplet._calc_height_and_grad(self)
        nodes = WaterDroplet._find_nodes_and_offsets(self.pos)[0]
        for node in nodes:
            weight = WaterDroplet._dist(self.pos, node) / np.sqrt(2)
            # if not self.d_h < 0:
            new_height = self.world.height_map[node[0]][node[1]] - np.abs(self.d_h) * weight * self.water
            if new_height < 0.05:
                self.world.height_map[node[0]][node[1]] = 0.05
            else:
                self.world.height_map[node[0]][node[1]] = new_height

    def erode_radius(self, radius=3.0):
        if not self.d_h < 0:
            weightings_dict = WaterDroplet._get_nodes_and_weights_in_radius(self.world.height_map, self.pos, radius)
            for pos, weight in weightings_dict.items():
                row, col = pos
                new_height = self.world.height_map[row][col] - (self.d_h * weight * self.water)
                if new_height < 0.05:
                    self.world.height_map[row][col] = 0.05
                else:
                    self.world.height_map[row][col] = new_height

    def evapourate(self):
        """Docstring"""
        if not self.water < 0:
            self.water -= self.EVAP_RATE

    def remove_drop(self):
        """Docstring"""
        pass

    @classmethod
    def change_inertia(cls, new_value):
        cls.INERTIA = new_value

    @classmethod
    def change_evapouration_rate(cls, new_value):
        cls.EVAP_RATE = new_value

    def update_height(self):
        pass

    def calc_gradient(self):
        pass

    def _calc_height_and_grad(self):
        """
        Given the world in which the drop exists and the current (x,y) position, this static method calculates the
        current height of the drop and the 2d gradient. This is required to determine the gravitational force on the
        drop when establishing its next position.
        """
        world = self.world.height_map
        nodes, offset = WaterDroplet._find_nodes_and_offsets(self.pos)
        height_nw = world[nodes[0][0]][nodes[0][1]]
        height_ne = world[nodes[1][0]][nodes[1][1]]
        height_sw = world[nodes[3][0]][nodes[3][1]]
        height_se = world[nodes[2][0]][nodes[2][1]]
        # Calculate droplet's direction of flow with bilinear interpolation of height difference along the edges
        x_grad = ((height_ne - height_nw) * (1 - offset[1])) + ((height_se - height_sw) * offset[1])
        y_grad = ((height_sw - height_nw) * (1 - offset[0])) + ((height_se - height_ne) * offset[0])
        # Calculate height with bilinear interpolation of the heights of the nodes of the cell
        self.z_pos = (height_nw * (1 - offset[0]) * (1 - offset[1])) + \
                     (height_ne * offset[0] * (1 - offset[1])) + \
                     (height_sw * (1 - offset[0]) * offset[1]) + \
                     (height_se * offset[0] * offset[1])
        gradient = np.array([x_grad, y_grad])

        return self.z_pos, gradient

    @staticmethod
    def _find_nodes_and_offsets(position_coord):
        x_coord = int(position_coord[0])
        y_coord = int(position_coord[1])
        # Calculate droplet's offset inside the cell (0,0) = at NW node, (1,1) = at SE node
        x_offset = position_coord[0] - x_coord
        y_offset = position_coord[1] - y_coord
        # Calculate heights of the four nodes of the droplet's cell
        x1 = x_coord
        x2 = x_coord + 1
        y1 = y_coord
        y2 = y_coord + 1
        nw = np.array([x1, y1])
        ne = np.array([x2, y1])
        se = np.array([x2, y2])
        sw = np.array([x1, y2])
        nodes = [nw, ne, se, sw]
        offset = np.array([x_offset, y_offset])
        return nodes, offset

    @staticmethod
    def _get_nodes_and_weights_in_radius(heightmap, drop_position, radius, funct='gauss'):
        x_minus, x_plus, y_minus, y_plus = WaterDroplet._calc_local_space(heightmap, drop_position, radius)
        dict_position_and_weights = {}
        for i in range(int(x_minus), int(x_plus)):
            for j in range(int(y_minus), int(y_plus)):
                grid_point = np.array([i, j])
                drop_to_gridpoint = WaterDroplet._dist(grid_point, drop_position)
                if drop_to_gridpoint < radius:
                    if funct == 'gauss':
                        weighting = WaterDroplet._gauss(drop_to_gridpoint, radius)
                        dict_position_and_weights[tuple(grid_point)] = weighting
        return dict_position_and_weights

    @staticmethod
    def _calc_local_space(heightmap, drop_position, radius):
        x, y = drop_position
        # Define subsection of map to search (+radius to -radius square)
        x_minus = np.ceil(x - radius) if np.ceil(x - radius) >= 0 else 0
        x_plus = np.floor(x + radius) if np.floor(x + radius) <= np.shape(heightmap)[0] - 1 else np.shape(heightmap)[0]
        y_minus = np.ceil(y - radius) if np.ceil(y - radius) >= 0 else 0
        y_plus = np.floor(y + radius) if np.floor(y + radius) <= np.shape(heightmap)[1] - 1 else np.shape(heightmap)[1]
        return x_minus, x_plus, y_minus, y_plus

    @staticmethod
    def _dist(pos1, pos2):
        distance = np.sqrt(np.sum((pos1 - pos2)**2))
        return distance

    @staticmethod
    def _gauss(distance, radius):
        return np.exp(-np.power(distance, 2) / (2 * np.power(np.sqrt(radius) / 2, 2)))


class RainCloud:
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

    def make_it_rain(self, strength=None):
        """ Creates an array of WaterDroplet objects and stores them in self.cloud list"""
        for _ in range(self.num_droplets):
            random_position = np.array([random.random() * self.world.lx, random.random() * self.world.ly])
            if strength:
                random_strength = strength[0] + (random.random() * strength[1])
                self.cloud.append(WaterDroplet(self.world, random_position, water=random_strength))
            else:
                self.cloud.append(WaterDroplet(self.world, random_position))
        return self.cloud

    def print_droplets(self):
        """Prints the position of each droplet in the cloud"""
        for drop in self.cloud:
            print(drop)

    def show_cloud(self):
        """Displays an image of the cloud"""
        print(f'This cloud has {self.num_droplets} droplets')
        cloud_image = np.zeros_like(self.world.height_map)
        for drop in self.cloud:
            i, j = drop.pos
            cloud_image[int(i), int(j)] = 1.0
        # scipy.misc.toimage(cloud_image).show()
        plt.figure()
        plt.imshow(cloud_image, cmap='gray')
        plt.show()


# To be run in the for testing and debugging purposes
if __name__ == '__main__':
    width, height = 100, 100
    w = earth.World(width, height)
    w.generate_height()

    test_dict = WaterDroplet._get_nodes_and_weights_in_radius(w.height_map, (45.2, 70.3), 6)
    print(test_dict)

    canvas = np.zeros(np.shape(w.height_map))
    for position, weight in test_dict.items():
        print(position, weight)
        i, j = position
        canvas[i][j] = weight

    plt.imshow(canvas, cmap='bone')
    plt.show()
