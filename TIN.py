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
        # Delaunay()

    def plotTriangulation(self):
        plt.triplot(self.x,  self.y, self.triangulation.simplices)
        plt.plot(self.x, self.y, 'o', markersize=2)
        plt.show()

    def findElevation(self, x, y):
        index = Delaunay.find_simplex(
            self.triangulation, [(x, y)], bruteforce=False, tol=None)

        p0, p1, p2 = self.triangulation.simplices[index][0]
        p0, p1, p2 = self.triangulation.points[p0], self.triangulation.points[p1], self.triangulation.points[p2]

    def plotAltitude(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=1)
        plt.show()


x = np.genfromtxt("pts1000c.dat")

myterrain = TIN(x)

myterrain.findElevation(0, 0)
# myterrain.plotTriangulation()
# myterrain.plotAltitude()
