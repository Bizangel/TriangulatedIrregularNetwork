from TIN import TIN
from perlin import PerlinNoiseFactory
import numpy as np

if __name__ == "__main__":
    # This is another test of suites,
    # with a relatively smooth (Perlin Noise) generated surface
    # Keep number of points low (less than 100)
    # so you can see results more clearly.

    # Number of sampled datapoints
    sz_datapoints = 100
    '''Data Point Creation'''
    PNFactory = PerlinNoiseFactory(2, octaves=1)

    X = np.zeros(sz_datapoints)
    Y = np.zeros(sz_datapoints)
    elevation = np.zeros(sz_datapoints)
    for i in range(sz_datapoints):
        x, y = np.random.random(2)*2 - 1  # Random points in -1, 1
        X[i], Y[i] = x, y
        elevation[i] = PNFactory(x, y)

    # Make elevations actually elevations (positive)
    elevation -= min(elevation)
    datapoints = np.column_stack([X, Y, elevation])

    '''TIN Testing'''
    # Create TIN object
    tin_net = TIN(datapoints)

    # Plot the elevation profile
    tin_net.plot_elevation_profile()

    # PT2: Find Elevation value of planar location
    x, y = np.random.random(2) * 0.5 - 1  # Random point in -0.5, 0.5 square
    # DISCLAIMER: point MIGHT not be inside triangulation
    # (should raise error correctly.)
    tin_net.find_elevation(x, y, display=True)

    # Report all peaks
    tin_net.plot_peaks()

    # Report all pits
    tin_net.plot_pits()

    # Report Triangulation and Dual Graph
    tin_net.plot_dual_graph(auto_show=True)
