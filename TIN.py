import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class TIN:
    '''
    Triangulated Irregular Network
    '''

    def __init__(self, datapoints):
        self.x = datapoints[:, 0]
        self.y = datapoints[:, 1]
        self.altitude = datapoints[:, 2]
        self.Initialize()

    def Initialize(self):
        points = np.column_stack((self.x, self.y))
        self.triangulation = Delaunay(points)
        print(points.shape)
        # Delaunay()

    def plotTriangulation(self):
        pass

    def plotAltitude(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=1)
        plt.show()


x = np.genfromtxt("pts1000c.dat")

print(x[:, 1].shape)
myterrain = TIN(x)

myterrain.plotAltitude()
