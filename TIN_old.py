from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

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

    def plotTriangulation(self, autoShow=True):
        fig = plt.figure()
        ax = plt.axes()
        ax.triplot(self.x,  self.y, self.triangulation.simplices)
        ax.plot(self.x, self.y, 'o', markersize=2)
        if autoShow:
            plt.show()
        return ax

    def findElevation(self, x, y):
        index = Delaunay.find_simplex(
            self.triangulation, [(x, y)], bruteforce=False, tol=None)

        p0, p1, p2 = self.triangulation.simplices[index][0]
        f0, f1, f2 = self.altitude[p0], self.altitude[p1], self.altitude[p2]
        p0, p1, p2 = self.triangulation.points[p0], self.triangulation.points[p1], self.triangulation.points[p2]

        A = np.column_stack([np.row_stack([p0, p1, p2]), np.ones(3)])
        try:
            a, b, c = np.matmul(np.column_stack(
                [f0, f1, f2]), np.linalg.inv(A))[0]
        except np.linalg.LinAlgError:
            raise ValueError(
                "Can't interpolate given point! No suitable 3d plane between triangulation exists!")
        # print(a, b, c)
        # f_interp = a*x + b*y + c
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=1)
        # print(x, y, f_interp)
        ax.plot()
        plt.show()

    def plotAltitude(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=1)
        plt.show()

    def plotDualGraph(self, autoShow=True):
        # Calculate the midpoint for each triangle (for display reasons)
        n_triangles = len(self.triangulation.simplices)
        midpoints = self.triangulation.points[self.triangulation.simplices]
        midpoints = np.average(midpoints, axis=1)

        # Calculate the pair of indices for each edge in the dual graph (according to the neighbors)
        x = self.triangulation.neighbors[range(n_triangles)]
        # print(x)
        tilehelp = np.tile(np.arange(n_triangles), (3, 1)).T
        # print(tilehelp)
        tilehelp = tilehelp.reshape((-1,))

        x = np.reshape(x, (n_triangles*3))
        pair_indexes = np.zeros(2*n_triangles*3, dtype='int32')
        pair_indexes[0::2] = tilehelp
        pair_indexes[1::2] = x
        pair_indexes = np.reshape(pair_indexes, (3*n_triangles, 2))

        # Remove rows with -1 as neighbor index
        pair_indexes = np.delete(pair_indexes, np.where(
            pair_indexes < 0), axis=0)

        # Remove repeated edges (There should only be one entry per edge)
        pair_indexes.sort(axis=1)
        pair_indexes = np.unique(pair_indexes, axis=0)

        lc = LineCollection(midpoints[pair_indexes],
                            linewidths=1, colors="green")
        ax = self.plotTriangulation(autoShow=False)
        ax.add_collection(lc)
        ax.scatter(midpoints[:, 0], midpoints[:, 1])
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

sz_datapoints = 7

X = np.zeros(sz_datapoints)
Y = np.zeros(sz_datapoints)
elevation = np.zeros(sz_datapoints)
for i in range(sz_datapoints):
    x, y = np.random.random(2)*2 - 1
    X[i], Y[i] = x, y
    elevation[i] = PNFactory(x, y)

elevation -= min(elevation)

datapoints = np.column_stack([X, Y, elevation])

myterrain = TIN(datapoints)

myterrain.plotDualGraph()
# print(myterrain.triangulation.convex_hull)

# print(len(myterrain.triangulation.simplices))
# print(len(myterrain.triangulation.neighbors))


# lc = mc.LineCollection(lines, linewidths=2, colors=colours)
# fig, ax = plt.subplots()
# ax.add_collection(lc)
# ax.autoscale()

# k = 1
# print(myterrain.triangulation.simplices[k])
# print(myterrain.triangulation.neighbors[k])

# indptr, indices = myterrain.triangulation.vertex_neighbor_vertices
# print(indices[indptr[k]:indptr[k+1]])

# myterrain.plotDualGraph()
# myterrain.findElevation(0, 0)
# myterrain.plotTriangulation()
# myterrain.plotAltitude()
