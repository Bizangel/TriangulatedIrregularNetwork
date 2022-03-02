import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from matplotlib.collections import LineCollection


class TIN:
    """Represents a TIN (Triangulated Irregular Network) of surface elevations.

    This class allows for several methods on a TIN, such as plotting,
    querying local maximums and minimums and plotting the Dual Graph of associated triangulation.
    """

    def __init__(self, datapoints: np.ndarray):
        """Initialize a TIN object out of a numpy ndarray.

        Parameters
        ----------
        datapoints : ndarray
            Numpy 2D array composed by three columns: x,y,z where z is the height.
            The datapoints to initialize and triangulate the TIN with.

        Raises
        ------
        ValueError
            Raised when an incompatible array is given as an input.
        """

        if (datapoints.ndim != 2):
            raise ValueError(
                "Incompatible Array Dimension in TIN object creation")
        if (datapoints.shape[1] != 3):
            raise ValueError(
                "Incompatible Array shape in TIN object creation, must be composed of three columns: x,y and z/height")
        self.x = datapoints[:, 0]
        self.y = datapoints[:, 1]
        self.altitude = datapoints[:, 2]
        if not np.all(self.altitude >= 0):
            raise ValueError(
                'Received negative altitude in TIN object creation.')

        points = np.column_stack((self.x, self.y))
        self.triangulation = Delaunay(points)
        self.triangulation.vertex_to_simplex

    def plot_triangulation(self, dot_size=0, auto_show=True, title="TIN Triangulation"):
        """Plot the 2D triangulation of the TIN.

        Creates a new matplotlib.pyplot Figure and Axes, and plots the 2D triangulation of the TIN.

        Parameters
        ----------
        dot_size : int, optional
            The size of the drawn vertex, if 0 vertex are not drawn, by default 0
        auto_show : bool, optional
            If true, calls plt.show() performing a blocking call to display the triangulation, by default True
        title : str, optional
            Optional title for the plot figure, by default TIN Triangulation

        Returns
        -------
        ~.axes.Axes
            A matplotlib.pyplot Axes object, with the 2D triangulation.
        """
        fig = plt.figure(title)
        ax = plt.axes()
        ax.triplot(self.x,  self.y, self.triangulation.simplices)
        if dot_size > 0:
            ax.plot(self.x, self.y, 'o', markersize=dot_size)
        if auto_show:
            plt.show()
        return ax

    def find_elevation(self, x: float, y: float, display=False):
        """Find the elevation of given point in the TIN surface.

        Finds the elevation of given point in the TIN surface, via linear interpolation.
        Raises an error if point is outside triangulation. 
        Optionally displays the point and the used triangle plane in interpolation.

        Parameters
        ----------
        x : float
            The X-coordinate of the point to find the elevation.
        y : float
            The Y-coordinate of the point to find the elevation.
        display : bool, optional
            If true, creates a new plt figure and displays the point and corresponding triangle, by default False

        Raises
        ------
        ValueError
            Raised if given point is outside triangulated surface.

        Returns
        -------
        float
            The elevation of the requested planar location.
        """
        index = Delaunay.find_simplex(
            self.triangulation, [(x, y)], bruteforce=False, tol=None)

        if index == -1:
            raise ValueError("Requested point not within Triangulation.")

        p0, p1, p2 = self.triangulation.simplices[index][0]
        f0, f1, f2 = self.altitude[p0], self.altitude[p1], self.altitude[p2]
        p0, p1, p2 = self.triangulation.points[p0], self.triangulation.points[p1], self.triangulation.points[p2]

        height = griddata([p0, p1, p2], [f0, f1, f2],
                          [x, y], method="linear")[0]
        if display:
            ax = self.plot_elevation_profile(autoShow=False, alpha=0.7)
            ax.plot([p0[0], p1[0], p2[0], p0[0]], [p0[1], p1[1], p2[1], p0[1]],
                    [f0, f1, f2, f0], marker=".", linewidth=2, c="red")
            ax.plot([x], [y], [height], marker="D", markersize=7, c="red")
            lab1 = mlines.Line2D([], [], color='red', marker='.', linestyle='None',
                                 markersize=10, label='Sample Points Used')
            lab2 = mlines.Line2D([], [], color='red', marker='D', linestyle='None',
                                 markersize=10, label='Interpolated Point')
            plt.legend(handles=[lab1, lab2])
            plt.show()
        return height

    def plot_local_maximum(self):
        """Plot the local maximum of the surface.

        Wrapper for plot_local_extremum.
        """
        self.plot_local_extremum(maximum=True)

    def plot_local_minimum(self):
        """Plot the local minimum of the surface.

        Wrapper for plot_local_extremum.
        """
        self.plot_local_extremum(maximum=False)

    def plot_peaks(self):
        """Plot the peaks of the given surface (local maximum).

        Alias of plot_local_maximum.
        """
        self.plot_local_maximum()

    def plot_pits(self):
        """Plot the pits of the given surface (local minimum).

        Alias of plot_local_minimum.
        """
        self.plot_local_minimum()

    def is_peak(self, pt_index: int):
        """Query wether a sample/input point is a peak (local maximum)

        Wrapper for is_local_extremum.

        Parameters
        ----------
        pt_index : int
            The index of the point to query, in the sample (input) points.

        Returns
        -------
        bool
            Whether the point is a peak.
        """
        return self.is_local_extremum(pt_index, maximum=True)

    def is_pit(self, pt_index: int):
        """Query wether a sample/input point is a pit (local minimum)

        Wrapper for is_local_extremum.

        Parameters
        ----------
        pt_index : int
            The index of the point to query, in the sample (input) points.

        Returns
        -------
        bool
            Whether the point is a pit.
        """
        return self.is_local_extremum(pt_index, maximum=False)

    def is_local_extremum(self, pt_index: int, maximum=True):
        """Query wether a sample point is a local extremum.

        A local minimum is a point that is lower than all of it's neighbors
        A local maximum is a point that is higher than all of it's neighbors

        Parameters
        ----------
        pt_index : int
            The index of the point to query, in the sample (input) points.
        maximum : bool, optional
            If true, will check for local maximum, else, will check for local minimum, by default True

        Returns
        -------
        bool
            Whether the point is a local extremum, according to parameters.
        """
        indptr, indices = self.triangulation.vertex_neighbor_vertices
        neighbor_vertices_indexes = indices[indptr[pt_index]                                            :indptr[pt_index+1]]
        is_extreme = True
        for neighbor in neighbor_vertices_indexes:
            if maximum:
                if self.altitude[neighbor] > self.altitude[pt_index]:
                    is_extreme = False
            else:
                if self.altitude[neighbor] < self.altitude[pt_index]:
                    is_extreme = False
        return is_extreme

    def plot_local_extremum(self, maximum=True):
        """Plot the either the local maximums or local minimums of a function.

        Parameters
        ----------
        maximum : bool, optional
            If True, will plot all local maximums, plots all local minimums otherwise, by default True
        """
        extremes = [i for i in range(
            len(self.altitude)) if self.is_local_extremum(i, maximum=maximum)]
        if maximum:
            maxminstring = "Local Maximum"
            color, mark = "blue", "*"
        else:
            maxminstring = "Local Minimum"
            color, mark = "red", "+"

        title = "TIN " + maxminstring
        ax = self.plot_elevation_profile(
            autoShow=False, alpha=0.7, title=title)
        for i in extremes:
            ax.plot(self.x[i], self.y[i], self.altitude[i],
                    c=color, marker=mark, markersize=7)
        lab = mlines.Line2D([], [], color=color, marker=mark, linestyle='None',
                            markersize=10, label=maxminstring)
        plt.legend(handles=[lab])
        plt.show()

    def plot_elevation_profile(self, autoShow=True, alpha=1, title="TIN Elevation Profile"):
        """Plot the associated surface to the TIN.

        Creates a new matplotlib.pyplot Figure and Axes, and plots the associated surface to the TIN.

        Parameters
        ----------
        auto_show : bool, optional
            If true, calls plt.show() performing a blocking call to display the triangulation, by default True

        alpha : int, optional
            The transparency of the plotted surface, by default 1

        title : str, optional
            An optional title for the plot figure, by default TIN Elevation Profile

        Returns
        -------
        ~.axes.Axes
            A matplotlib.pyplot Axes object, with 3D projection and the plotted surface.
        """
        fig = plt.figure(title)
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(self.x, self.y, self.altitude,
                        color='white', cmap="BrBG", alpha=alpha)
        if autoShow:
            plt.show()
        return ax

    def plot_dual_graph(self, auto_show=True, dot_size=2.0):
        """Plot the associated dual graph of the TIN triangulation.

        Parameters
        ----------
        auto_show : bool, optional
            If true, calls plt.show() performing a blocking call to display the triangulation, by default True
        dot_size : int, optional
            The size of the drawn graph vertex

        Returns
        -------
         ~.axes.Axes
            A matplotlib.pyplot Axes object, with the dual graph on top of the triangulation 
        """
        # Calculate the midpoint for each triangle (for display reasons)
        n_triangles = len(self.triangulation.simplices)
        midpoints = self.triangulation.points[self.triangulation.simplices]
        midpoints = np.average(midpoints, axis=1)

        # Calculate the pair of indices for each edge in the dual graph (according to the neighbors)
        x = self.triangulation.neighbors[range(n_triangles)]
        tilehelp = np.tile(np.arange(n_triangles), (3, 1)).T
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
        ax = self.plot_triangulation(
            auto_show=False, title="TIN Dual Graph")
        ax.add_collection(lc)
        ax.scatter(midpoints[:, 0], midpoints[:, 1], s=dot_size)
        lab = mlines.Line2D([], [], color="blue",
                            markersize=10, label="Triangulation")
        lab2 = mlines.Line2D([], [], color="green",
                             markersize=10, label="Dual Graph")
        plt.legend(handles=[lab, lab2])
        if auto_show:
            plt.show()
        return ax
