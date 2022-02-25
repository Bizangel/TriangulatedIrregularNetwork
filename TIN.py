import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from perlin import PerlinNoiseFactory


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


'''Read from input'''
# x = np.genfromtxt("pts1000c.dat")

'''Generate Input: '''
PNFactory = PerlinNoiseFactory(2, octaves=1)

'''Evenly sampled generation'''
# frameSize = 10
# X, Y = np.meshgrid(np.linspace(-2, 2, frameSize),
#                    np.linspace(-2, 2, frameSize))

# noise = np.zeros([frameSize, frameSize])
# for i in range(frameSize):
#     for j in range(frameSize):
#         noise[i, j] = PNFactory(i/frameSize, j/frameSize)


# elevation = np.reshape(noise, (-1,))
# datapoints = np.column_stack([X, Y, np.reshape(noise, (-1,))])

'''Non-evenly sampled generation'''

sz_datapoints = 200

X = np.zeros(sz_datapoints)
Y = np.zeros(sz_datapoints)
elevation = np.zeros(sz_datapoints)
for i in range(sz_datapoints):
    x, y = np.random.random(2)*2 - 1
    X[i], Y[i] = x, y
    elevation[i] = PNFactory(x, y)

datapoints = np.column_stack([X, Y, elevation])

myterrain = TIN(datapoints)


# myterrain.findElevation(0, 0)
# myterrain.plotTriangulation()
# myterrain.plotAltitude()
