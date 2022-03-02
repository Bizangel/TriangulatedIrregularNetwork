from TIN import TIN
import numpy as np

if __name__ == "__main__":
    # Do also check main2.py, due to the high number of points,
    # it might not be easy to see all the features of the TIN implementation.
    # Initialization of TIN object
    datapoints = np.genfromtxt("pts1000c.dat")
    tin_net = TIN(datapoints)

    # PT1: Plotting the elevation profile
    tin_net.plot_elevation_profile()

    # PT2: Find Elevation value of planar location
    x, y = np.random.random(2) * 20 - 10  # Random point in -10, 10 square
    height = tin_net.find_elevation(x, y, display=True)
    print(f"Interpolated Height: {height} of point ({x},{y})")

    # PT3: a) Query Whether a point is a water source.
    # Peak point in sample data or relative maximum
    pt_index = np.random.randint(len(datapoints))  # Get a random sample point
    print(f"Is ({x},{y}) a maximum?: {tin_net.is_peak(pt_index)}")
    # PT3: b) Report all peaks.
    tin_net.plot_peaks()

    # PT4: a) Query Whether a point is a water pit.
    # Pit point in sample data or relative minimum
    print(f"Is ({x},{y}) a minimum?: {tin_net.is_pit(pt_index)}")
    # PT4: b) Report all peaks.
    tin_net.plot_pits()

    # PT5 Construct and plot the dual graph of the triangulated terrain:
    # Make sure to zoom in, due to a lot of points!
    tin_net.plot_triangulation(auto_show=False)
    tin_net.plot_dual_graph(auto_show=True)
