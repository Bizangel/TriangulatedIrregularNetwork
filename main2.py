from perlin import PerlinNoiseFactory

if __name__ == "__main__":
    '''Generate Input: '''
    PNFactory = PerlinNoiseFactory(2, octaves=1)

    '''Non-evenly sampled generation'''

    # sz_datapoints = 100

    # X = np.zeros(sz_datapoints)
    # Y = np.zeros(sz_datapoints)
    # elevation = np.zeros(sz_datapoints)
    # for i in range(sz_datapoints):
    #     x, y = np.random.random(2)*2 - 1
    #     X[i], Y[i] = x, y
    #     elevation[i] = PNFactory(x, y)

    # make elevations actually elevations (positive)
    # elevation -= min(elevation)

    # datapoints = np.column_stack([X, Y, elevation])

    # myterrain = TIN(datapoints)

    # myterrain.plotPeaks()
    # myterrain.plotPits()

    # myterrain.plotDualGraph()

    # myterrain.plotTriangulation()

    # print(myterrain.triangulation.vertex_neighbor_vertices)
    # myterrain.findElevation(*(np.random.random(2)*2 - 1))
    # myterrain.plotTriangulation()
    # myterrain.plotAltitude()
