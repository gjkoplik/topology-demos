"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
make_vector_fields.py

Create vector fields and interpolate data to them
"""

import numpy as np
from scipy.interpolate import griddata
from sklearn.cluster import KMeans

def vf_2_sinks(num_x=20, num_y=20, xbounds=(0, 10), ybounds=(0, 10), center_0=None, center_1=None):
    """
    create vector field of two sinks

    :param num_x (int, default 20): number of vectors along the x dimension
    :param num_y (int, default 20): number of vectors along the y dimension
    :param xbounds (tuple of ints, default (0, 10): span of vectors along x dimension
    :param ybounds (tuple of ints, default (0, 10): span of vectors along y dimension
    :param center_0 (np.array of dim (2,), default `None`): location of first center
        (`None` results in random initialization inside bounds)
    :param center_1 (np.array of dim (2,), default `None`): location of second center
        (`None` results in random initialization inside bounds)
    :return X, Y, U, V, center_0, center_1:
        X (np.array) X coords of vectors (as single column)
        Y (np.array) Y coords of vectors (as single column)
        U (np.array) X component of each vector (as single column)
        V (np.array) Y component of each vector (as single column)
        center_0 (np.array) first center (as single column)
        center_1 (np.array) second center (as single column)
    """

    # random centers if none specified
    if center_0 is None:
        center_0 = np.array([np.random.uniform(xbounds[0], xbounds[1]),
                             np.random.uniform(ybounds[0], ybounds[1])])
    if center_1 is None:
        center_1 = np.array([np.random.uniform(xbounds[0], xbounds[1]),
                             np.random.uniform(ybounds[0], ybounds[1])])
    xspace = np.linspace(xbounds[0], xbounds[1], num_x)
    yspace = np.linspace(ybounds[0], ybounds[1], num_y)
    X, Y = np.meshgrid(xspace, yspace)

    # distance from X,Y points to each center
    #   if positive, closer to center_0, else closer to center_1
    closer_to_0 = (np.sqrt((X - center_0[0]) ** 2 + (Y - center_0[1]) ** 2)
                   - np.sqrt((X - center_1[0]) ** 2 + (Y - center_1[1]) ** 2)
                   <= 0).flatten()

    center_pointing = np.ones((X.size, 2)) * center_1
    center_pointing[np.where(closer_to_0), :] = center_0

    # get normed vectors pointing to each center
    U_V_vectors = np.column_stack((center_pointing[:, 0] - X.reshape((-1,)),
                                   center_pointing[:, 1] - Y.reshape((-1,))))
    # normalize
    norm = np.linalg.norm(U_V_vectors, axis=1)

    U = U_V_vectors[:, 0] / norm
    V = U_V_vectors[:, 1] / norm

    # flatten all of them for interpolation later
    X = X.reshape((-1,))
    Y = Y.reshape((-1,))
    U = U.reshape((-1,))
    V = V.reshape((-1,))
    return X, Y, U, V, center_0, center_1

def vf_3_loops(num_x=20, num_y=20,
               center_0=None, center_1=None, center_2=None,
               radii=np.array([2, 4, 6]),
               xbounds=(0, 10), ybounds=(0, 10)):
    """
    create vector field of loop of points around center of bounds

    :param num_x (int, default 20): number of vectors along the x dimension
    :param num_y (int, default 20): number of vectors along the y dimension
    :param center_0 (np.array of dim (2,), default `None`): location of first loop center
        (`None` results in random initialization inside bounds)
    :param center_1 (np.array of dim (2,), default `None`): location of second loop center
        (`None` results in random initialization inside bounds)
    :param center_2 (np.array of dim (2,), default `None`): location of third loop center
        (`None` results in random initialization inside bounds)
    :param radii (int > 0, default 5): radius of loop to force points toward
    :param xbounds (tuple of ints, default (0, 10): span of vectors along x dimension
    :param ybounds (tuple of ints, default (0, 10): span of vectors along y dimension
    :return X, Y, U, V:
        X (np.array) X coords of vectors (as single column)
        Y (np.array) Y coords of vectors (as single column)
        U (np.array) X component of each vector (as single column)
        V (np.array) Y component of each vector (as single column)
    """

    # random centers if none specified
    if center_0 is None:
        center_0 = np.array([np.random.uniform(xbounds[0], xbounds[1]),
                             np.random.uniform(ybounds[0], ybounds[1])])
    if center_1 is None:
        center_1 = np.array([np.random.uniform(xbounds[0], xbounds[1]),
                             np.random.uniform(ybounds[0], ybounds[1])])
    if center_2 is None:
        center_2 = np.array([np.random.uniform(xbounds[0], xbounds[1]),
                             np.random.uniform(ybounds[0], ybounds[1])])
    # build voronoi regions via k-means clustering with 3 points
    points = np.vstack((center_0, center_1, center_2))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit_transform(points)
    xspace = np.linspace(xbounds[0], xbounds[1], num_x)
    yspace = np.linspace(ybounds[0], ybounds[1], num_y)
    X, Y = np.meshgrid(xspace, yspace)
    # figure out each vector's assigned cell
    Z = kmeans.predict(np.c_[X.flatten(), Y.flatten()])
    # create empty arrays of U, V values to add to
    U = np.array([])
    V = np.array([])
    # and their corresponding X and Y values
    X_sorted = np.array([])
    Y_sorted = np.array([])
    # create loop vf within each voronoi region
    for i, center in enumerate([center_0, center_1, center_2]):
        X_temp = X.flatten()[np.where(Z == kmeans.labels_[i])]
        Y_temp = Y.flatten()[np.where(Z == kmeans.labels_[i])]
        U_temp = X_temp - center[0]
        V_temp = Y_temp - center[1]
        # flip vectors
        to_flip = np.square(X_temp - center[0]) + \
                  np.square(Y_temp - center[1]) > np.square(radii[i])
        U_temp[np.where(to_flip)] = -U_temp[np.where(to_flip)]
        V_temp[np.where(to_flip)] = -V_temp[np.where(to_flip)]
        # append all values to relevant whole vector field arrays
        U = np.append(U, U_temp)
        V = np.append(V, V_temp)
        X_sorted = np.append(X_sorted, X_temp)
        Y_sorted = np.append(Y_sorted, Y_temp)

    # get normed vectors pointing to each center
    U_V_vectors = np.column_stack((U, V))
    # normalize
    norm = np.linalg.norm(U_V_vectors, axis=1)

    U = U_V_vectors[:, 0] / norm
    V = U_V_vectors[:, 1] / norm

    # flatten all of them for interpolation later
    U = U.reshape((-1,))
    V = V.reshape((-1,))
    return X_sorted, Y_sorted, U, V

def interpolate_vector_field_to_data(underlying_vector_field, data):
    """
    find vectors for self.data via bilinear interpolation
    between known vectors in underlying_vector_field

    :param underlying_vector_field (tuple of X, Y, U, V np.arrays):
    :param data ((n, 2) np.array): data on which to interpolate
    :return:
    """
    X, Y, U, V = underlying_vector_field
    x_interp_space = data[:, 0].reshape((-1,))
    y_interp_space = data[:, 1].reshape((-1,))
    points = np.hstack((X.reshape((-1, 1)), Y.reshape((-1, 1))))
    x_component = griddata(points=points, values=U, xi=(x_interp_space, y_interp_space))
    y_component = griddata(points=points, values=V, xi=(x_interp_space, y_interp_space))
    return x_component, y_component

def update_data_location(data, interpolated_vector_field, update_scalar=0.2):
    """
    update the location of self.data
        as a scaled function of self.interpolated_vector_field

    :param data ((n, 2) np.array): data to update
    :param interpolated_vector_field (output from `interpolate_vector_field_to_data()`)
    :param update_scalar (float, default=1): value that scales how much to update points
    :return:
    """
    data[:, 0] += update_scalar * interpolated_vector_field[0]
    data[:, 1] += update_scalar * interpolated_vector_field[1]

    return data