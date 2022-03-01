import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
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
        assert np.all(self.altitude >= 0), 'Negative altitudes!'
        self.Initialize()

    def Initialize(self):
        points = np.column_stack((self.x, self.y))
        self.triangulation = Delaunay(points)
        # Delaunay()

    def plotTriangulation(self, dotSize=3, autoShow=True):
        fig = plt.figure()
        ax = plt.axes()
        ax.triplot(self.x,  self.y, self.triangulation.simplices)
        ax.plot(self.x, self.y, 'o', markersize=dotSize)
        if autoShow:
            plt.show()
        return ax

    def findElevation(self, x, y):
        index = Delaunay.find_simplex(
            self.triangulation, [(x, y)], bruteforce=False, tol=None)

        if index == -1:
            raise ValueError("Not within map")

        p0, p1, p2 = self.triangulation.simplices[index][0]
        f0, f1, f2 = self.altitude[p0], self.altitude[p1], self.altitude[p2]
        p0, p1, p2 = self.triangulation.points[p0], self.triangulation.points[p1], self.triangulation.points[p2]

        height = griddata([p0, p1, p2], [f0, f1, f2],
                          [x, y], method="linear")[0]

        ax = self.plotAltitude(autoShow=False, alpha=0.7)
        ax.plot([p0[0], p1[0], p2[0], p0[0]], [p0[1], p1[1], p2[1], p0[1]],
                [f0, f1, f2, f0], marker=".", linewidth=2, c="red")
        ax.plot([x], [y], [height], marker="D", markersize=7, c="red")
        plt.show()

    def plotPeaks(self):
        self.plotExtremes(peaks=True)

    def plotPits(self):
        self.plotExtremes(peaks=False)

    def plotExtremes(self, peaks=True):

        indptr, indices = self.triangulation.vertex_neighbor_vertices
        extremes = []

        # print(indices)
        for i in range(len(self.triangulation.points)):
            neighbor_vertices_indexes = indices[indptr[i]:indptr[i+1]]
            # print(i, "-->", neighbor_vertices_indexes)
            isExtreme = True
            for neighbor in neighbor_vertices_indexes:
                if peaks:
                    if self.altitude[neighbor] > self.altitude[i]:
                        isExtreme = False
                else:
                    if self.altitude[neighbor] < self.altitude[i]:
                        isExtreme = False
            if isExtreme:
                extremes.append(i)

        # print(extremes)
        ax = self.plotAltitude(autoShow=False, alpha=0.7)
        for i in extremes:
            # print(f'Point: {self.x[i]}, {self.y[i]}, {self.altitude[i]}')
            ax.plot(self.x[i], self.y[i], self.altitude[i],
                    c="blue", marker="*", markersize=7)
        plt.show()
        # xs = self.x[extremes]
        # altitudes = self.altitude[extremes]

    def plotAltitude(self, autoShow=True, alpha=1):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=alpha)
        if autoShow:
            plt.show()
        return ax

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

'''Non-evenly sampled generation'''

sz_datapoints = 100

X = np.zeros(sz_datapoints)
Y = np.zeros(sz_datapoints)
elevation = np.zeros(sz_datapoints)
for i in range(sz_datapoints):
    x, y = np.random.random(2)*2 - 1
    X[i], Y[i] = x, y
    elevation[i] = PNFactory(x, y)

# make elevations actually elevations (positive)
elevation -= min(elevation)

datapoints = np.column_stack([X, Y, elevation])

myterrain = TIN(datapoints)


# myterrain.plotPeaks()
myterrain.plotPits()

# myterrain.plotDualGraph()

# myterrain.plotTriangulation()


# print(myterrain.triangulation.vertex_neighbor_vertices)
# myterrain.findElevation(*(np.random.random(2)*2 - 1))
# myterrain.plotTriangulation()
# myterrain.plotAltitude()
