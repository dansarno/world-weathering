import noise
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# from mayavi import mlab
import random


class World:
    # add class variables here
    def __init__(self, lx, ly):
        """
        Constructor method
        Args:
            lx:
            ly:
        """
        self.lx = lx
        self.ly = ly
        self.height_map = np.zeros((lx, ly))

    def __repr__(self):
        """Returns representation of the object e.g. World(100, 100)"""
        return f'{self.__class__.__name__}({self.lx}, {self.ly})'

    def __str__(self):
        """Returns a human-readable string describing the world object"""
        return f'This world is {self.lx} x {self.ly} pixels'

    def __len__(self):
        """Returns the length of the height map array i.e. total num of pixels"""
        return len(self.height_map)

    def __getitem__(self, item):
        # TODO write some method which can allow for 2d array indexing i.e. world[i][j]
        pass

    # TODO need to consider the effect of scale and how it propagates through the project
    def generate_height(
                        self,
                        scale=40.0,
                        octaves=4,
                        persistence=0.5,
                        lacunarity=2.0,
                        rand=True,
                        xoff=0.0,
                        yoff=0.0
                        ):
        """
        Uses 2d perlin noise to generate a "terrain" where the value of each array element signifies the height
        Args:
            scale:
            octaves:
            persistence:
            lacunarity:
            rand:
            xoff:
            yoff:

        Returns:

        """
        # Choose from one of 10,000 random landscapes
        if rand:
            xoff = random.randint(1, 100)
            yoff = random.randint(1, 100)

        for i in range(self.lx):
            for j in range(self.ly):
                self.height_map[i][j] = noise.pnoise2((i / scale) + xoff,
                                                      (j / scale) + yoff,
                                                      octaves=octaves,
                                                      persistence=persistence,
                                                      lacunarity=lacunarity
                                                      )
        # Normalisation - not sure if needed
        self.height_map = (self.height_map - np.min(self.height_map))/np.ptp(self.height_map)

    def add_hardness_map(self):
        pass

    def show_map(self):
        """

        Returns: N/A

        """
        plt.figure()
        plt.imshow(self.height_map, cmap="gray")

    def matplotlib_plot(self):
        """

        Returns:

        """
        xx, yy = World._generate_grid(self, transpose_return=False)
        zz = self.height_map
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # def mayavi_plot(self, new_fig=False, cmap='copper', scale=400):
    #     """
    #     """
    #     xx, yy = World._generate_grid(self)
    #     if new_fig:
    #         mlab.figure(size=(800, 640))
    #     mlab_obj = mlab.surf(xx, yy, self.height_map, colormap=cmap, warp_scale=scale)
    #     return mlab_obj

    def _generate_grid(self, transpose_return=True):
        """ Private method to generate a grid given the world dimensions"""
        x = np.linspace(1, self.lx, self.lx)
        y = np.linspace(1, self.ly, self.ly)
        xx, yy = np.meshgrid(x, y)
        if transpose_return:
            return xx.T, yy.T
        else:
            return xx, yy


# To be run in the for testing and debugging
if __name__ == '__main__':
    width, height = 1000, 1000
    # Generate 5 random landscapes
    for f_t in range(2):
        w = World(width, height)
        w.generate_height(scale=400, rand=bool(f_t))
        w.show_map()
        # w.mayavi_plot(new_fig=True, scale=400)

    # plt.imshow(canvas, cmap='bone')
    # mlab.show()
