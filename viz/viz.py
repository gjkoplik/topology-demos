"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
viz.py

viz functions to be used in runners
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import matplotlib.colors as pltc
from matplotlib.colors import to_rgba_array
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os

def plot_persistence(pers_data,
                     bounds=None,
                     title='Persistence Diagram',
                     save=False, path='./figures/',
                     name='persistence.png', show=True, dpi=150,
                     figsize=(6, 6),
                     color='C0',
                     fig=None, ax=None, alpha=0.5):
    """
    plot persistence diagram

    :param pers_data (pd df): df of persistence information
    :param bounds (tuple of ints, default `None`): (min, max) x and y values for figure
        if `None`, will infer shape by max persistence value
    :param title (str, default 'Persistence Diagram'): title of resulting figure
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default 'persistence.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :param figsize (tuple of positive floats, default (6, 4) ): (horizontal, vertical) dimensions of figure
    :param color (str, default 'C0'): color of the persistence values in the figure
    :param fig (matplotlib fig): defaults to None (makes own fig)
    :param ax (matplotlib ax): defaults to None (makes own ax)
    :param alpha (float in [0, 1]): transparancy value for points
    :return fig, ax:
    """

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # build 45 degree line


    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.plot([0, bounds[1]], [0, bounds[1]], c='black')
    else:
        max_death = pers_data.death.values.max()
        min_death = min(0, pers_data.death.values.min())
        min_birth = min(0, pers_data.birth.values.min())
        min_val = min(min_birth, min_death)
        ax.plot([min_val, max_death], [min_val, max_death], c='black')

    if pers_data.shape[0] != 0:
        ax.scatter(pers_data.birth, pers_data.death,
                   alpha=alpha, c=color)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    ax.set_title(title)
    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax

def pers_0d_disk_viz(data, pers, radius,
                     disk_colors=0, buffer=3,
                     vmin=None, vmax=None, cmap=None,
                     save=False, path='./figures/',
                     name='0d_pers_viz.png', show=True, dpi=150):
    """
    visualize growing balls around 2d data while building out persistence diagram simultaneously

    :param data (np.array shape (n, 2)): points to plot
    :param pers (pd.DataFrame): 0d persistence info as called by
        `pc = multidim.PointCloud(data, max_length=-1).make_pers0()` then `pc.pers0.diagram`
    :param radius (float > 0): radius to grow balls and 0d persistence at or below which to plot
    :param disk_colors (float or list of floats, default 0): float values for color for disk
        if single float input for color, will use same color for all disks
        if list input for color, will expect color for every data point
    :param buffer (float > 0, default 3): buffer area around data for data figure on ax1
    :param vmin (float, default `None`): min value on colormap
        default `None` uses min value of disk_colors
    :param vmax (float, default `None`): max value on colormap
        default `None` uses max value of disk_colors
    :param cmap (`matplotlib.pyplot.cm` colormap, default `None`): cmap to use for disks
        default `None` uses `matplotlib.pyplot.cm.tab10`
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default '0d_pers_viz.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return fig, ax1, ax2:
    """

    # coax list input to 1d array
    if type(disk_colors) == list:
        disk_colors = np.array(disk_colors)
    # color for every point
    if type(disk_colors) == np.ndarray:
        assert disk_colors.shape[0] == data.shape[0]
    # coax float to array of colors for each point
    if type(disk_colors) != list and type(disk_colors) != np.ndarray:
        disk_colors = np.array([disk_colors] * data.shape[0])

    max_radius = pers.loc[:, 'pers'].values.max()

    pers_subset = pers.loc[pers.death <= radius, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.axis('equal')
    ax2.axis('equal')
    ax1.set_xlim(data[:, 0].min() - buffer, data[:, 0].max() + buffer)
    ax1.set_ylim(data[:, 1].min() - buffer, data[:, 1].max() + buffer)
    ax1.set_title('Growing Disks Around Each Point')

    # balls
    if cmap is None:
        cmap = plt.cm.tab10
    patches = []
    for i in range(data.shape[0]):
        circle = mpatches.Circle(data[i, :], radius)
        patches.append(circle)
    p = PatchCollection(patches, alpha=0.9, cmap=cmap)
    p.set_array(np.array(disk_colors))
    ax1.add_collection(p)
    if vmin is None:
        vmin = np.array(disk_colors).min()
    if vmax is None:
        vmax = np.array(disk_colors).max()
    p.set_clim(vmin, vmax)
    # data
    ax1.scatter(data[:, 0], data[:, 1], c='C1', edgecolor='black')

    # adding to persistence diagram as we go
    ax2.axhline(y=radius, c='red', label='Current Radius {:.2f}'.format(radius))
    ax2.legend(loc='lower right')
    plot_persistence(pers_subset, fig=fig, ax=ax2, alpha=0.7,
                     bounds=(-0.1, max_radius + 2), show=False)
    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax1, ax2


def pers_with_data_viz(data, pers, pers_min=-0.1, pers_max=None,
                       data_xrange=None, data_yrange=None,
                       save=False, path='./figures/',
                       name='pers_with_data_viz.png', show=True, dpi=150):
    """
    visualize 2d data with entire persistence diagram

    :param data (np.array shape (n, 2)): points to plot
    :param pers (pd.DataFrame): persistence info as called by
        `pc = multidim.PointCloud(data, max_length=-1).make_pers0()` then `pers = pc.pers0.diagram` or
        `pc = multidim.PointCloud(data, max_length=-1).make_pers1_rca1()` then `pers = pc.pers1.diagram`
    :param pers_min (float, default -0.1): min y value to show for persistence diagram
    :param pers_max (float, default `None`): max y value to show for persistence diagram
        default `None` sets at max death value in `pers` plus 1
    :param data_xrange (tuple of two floats, default `None`): x range for data plot
        default `None` sets range at `(data[:, 0].min() - 1, data[:, 0].max() + 1)`
    :param data_yrange (tuple of two floats, default `None`): y range for data plot
        default `None` sets range at `(data[:, 1].min() - 1, data[:, 1].max() + 1)`
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default 'pers_with_data_viz.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return fig, ax1, ax2:
    """

    if pers_max is None:
        pers_max = pers.loc[:, 'death'].values.max() + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.axis('equal')
    ax2.axis('equal')
    if data_xrange is None:
        ax1.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    else:
        ax1.set_xlim(data_xrange[0], data_xrange[1])
    if data_yrange is None:
        ax1.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    else:
        ax1.set_ylim(data_yrange[0], data_yrange[1])
    ax1.set_title('Data')

    # add points
    ax1.scatter(data[:, 0], data[:, 1], c='C0', edgecolor='black')

    # adding persistence diagram
    plot_persistence(pers, fig=fig, ax=ax2, alpha=0.7, bounds=(pers_min, pers_max), show=False)
    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax1, ax2

def plot_vector_field(X, Y, U, V,
                      save=False, path='./figures/',
                      name='underlying_vector_field.png', show=True, dpi=150):
    """
    plot vector field

    :param X (np.array): X coord of each vector
    :param Y (np.array): Y coord of each vector
    :param U (np.array): X component of each vector
    :param V (np.array): Y component of each vector
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default 'underlying_vector_field.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return fig, ax:
    """
    X_range = X.max() - X.min()
    Y_range = Y.max() - Y.min()
    fig, ax = plt.subplots(figsize=(6, 6 * (Y_range / X_range)))
    ax.set_title('Underlying Vector Field')
    ax.quiver(X, Y, U, V)
    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax

def plot_3d_surface(X, Y, Z, angle_above_XY, rotation_XY, cmap='viridis',
                    include_colorbar=True,
                    save=False, path='./figures/',
                    name='3d_surface.png', show=True, dpi=150):
    """
    plot 3d surface of data at specific angle

    :param X (np.array): X mesh grid
    :param Y (np.array): Y mesh grid
    :param Z (np.array): heights corresponding to X and Y indices
    :param angle_above_XY (float): angle above the XY plane to plot figure
    :param rotation_XY (float): angle rotating around Z axis
    :param cmap (str, default 'viridis'): color scheme for surface according to height
    :param include_colorbar (bool, default True): whether or not to include the colorbar
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default '3d_surface.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return fig, ax:
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(angle_above_XY, rotation_XY)
    surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, antialiased=False)
    if include_colorbar:
        cb = fig.colorbar(surface)
        cb.ax.set_title('Height Color')
    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax


def categorical_cmap(threshold, base_cmap='viridis', vmin=0, vmax=1, below_color=None,
                     above_color=None):
    """
    build categorical cmap to represent accessible regions coloring above and / or below threshold

    :param threshold (float): cutoff in cmap where we get categorical change
    :param base_cmap (str, default 'viridis'): base cmap to use
    :param vmin (float, default 0): min value in cmap (for scaling `threshold`)
    :param vmax (float, default 1): max value in cmap (for scaling `threshold`)
    :param below_color (str, default `None`): color below threshold
        default `None` leaves below threshold color unchanged
    :param above_color (str, default `None`): color above threshold
        `default `None` leaves above threshold color
    :return matplotlib.colors.ListedColormap:

    """
    assert threshold <= vmax, "param `threshold` must be less than `vmax`"
    assert threshold >= vmin, "param `threshold` must be greater than `vmin`"

    # actual cmap as a baseline (will overwrite the part we care about)
    cmap = cm.get_cmap(base_cmap, 256)
    newcolors = cmap(np.linspace(0, 1, 256))
    # figure out where threshold is relative to vmin, vmax
    percentage_cutoff = (threshold - vmin) / (vmax - vmin)
    int_cutoff = int(256 * percentage_cutoff)
    if below_color is not None:
        newcolors[:int_cutoff, :] = to_rgba_array(below_color)
    if above_color is not None:
        newcolors[int_cutoff:, :] = to_rgba_array(above_color)

    return ListedColormap(newcolors)


def random_cmap(array, indices_to_change=None, change_color='white'):
    """
    build a matplotlib colormap for values in `array` (one random color per element of `array`)

    :param array (np.array): (n, ) np.array for which colors will span
    :param indices_to_change (bool, default True): whether to create an extra index at end associated to `change_color`
    :param change_color (str, default 'white'): color to change extra index in resulting cmap
    :return matplotlib.colors.ListedColormap:
    """

    # one random color for each element in array
    cmap_size = array.size

    # get random color names
    all_colors = [k for k, v in pltc.cnames.items()]
    colors = np.random.choice(all_colors, cmap_size)

    # build empty cmap
    newcolors = np.empty((cmap_size, 4))

    # fill in with random colors in rgba form
    for i, color in enumerate(colors):
        newcolors[i, :] = to_rgba_array(color)

    if indices_to_change is not None:
        newcolors = np.r_[newcolors, to_rgba_array(change_color)]

    return ListedColormap(newcolors)

def plot_image(image, cmap='viridis',
               fig=None, ax=None,
               origin='low',
               title='3d Scenario', cb_title='Height',
               include_colorbar=True,
               save=False, path='./figures/',
               name='image.png', show=True, dpi=150):
    """
    plot a numpy array

    :param image ((m, n) np.array):
    :param cmap (str, default 'viridis'): cmap to use
    :param fig (matplotlib fig): defaults to None (makes own fig)
    :param ax (matplotlib ax): defaults to None (makes own ax)
    :param origin (str, default 'low'): where origin should be on image
    :param title (str, default '3d Scenario'): title of figure
    :param cb_title (str, default 'Height'): title of color bar
    :param include_colorbar (bool, default True): whether or not to include the colorbar
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default 'image.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return fig, ax:
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    a = ax.imshow(image, origin=origin, cmap=cmap)
    ax.axis('off')
    ax.set_title(title)
    if include_colorbar:
        cb = fig.colorbar(a)
        cb.ax.set_title(cb_title)

    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax

def plot_signal_threshold(data, tau,
                          fig=None, ax=None,
                          tau_color='black', data_start_color='gray',
                          below_tau_color='C0', above_tau_alpha=0.2,
                          figsize=(6, 4), title='Signal',
                          save=False, path='./figures/',
                          name='signal.png', show=True, dpi=150):
    """
    plot signal data at sweeping threshold `tau`

    :param data ( (n, 2) np.array of floats): x and y coords of signal data points
    :param tau (float): value up to which we have swept up to from below the signal
    :param fig (matplotlib fig): defaults to None (makes own fig)
    :param ax (matplotlib ax): defaults to None (makes own ax)
    :param tau_color (str, default 'C3'): color for sweep line
    :param data_start_color (str, default 'gray'): color for data above tau
    :param below_tau_color (str or list, default 'C0'): color for data at or below tau
        if single str input for color, will use same color for all points and lines below
        if list input for color, will expect color for every data point
    :param above_tau_alpha (float in [0, 1], default 0.2): alpha value for signal above tau
    :param figsize (tuple of positive floats, default (6, 4) ): (horizontal, vertical) dimensions of figure
    :param title (str, default 'Signal'): title of figure
    :param save (bool, default False): whether or not to save the plot
    :param path (str, default './figures/): path to where to save plot
    :param name (str, default 'image.png): name of png file to save
    :param show (bool, default True): whether or not to show the plot
    :param dpi (int > 0, default 150): pixel density to save figure at
    :return:
    """

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # coax 1d array input to list
    if type(below_tau_color) == np.ndarray:
        below_tau_color = list(below_tau_color)
    # color for every point
    if type(below_tau_color) == list:
        assert len(below_tau_color) == data.shape[0]
    # coax string to list of colors for each point\
    if type(below_tau_color) == str:
        below_tau_color = [below_tau_color] * data.shape[0]

    # logic to handle different color above tau vs below tau
    for i in range(data.shape[0] - 1):
        # slope between two points
        rise = data[i + 1, 1] - data[i, 1]
        run = data[i + 1, 0] - data[i, 0]
        # how much to scale slope to hit intersection
        scale = (tau - data[i, 1]) / rise
        delta_x = scale * run
        # both above tau
        if data[i, 1] > tau and data[i + 1, 1] > tau:
            ax.scatter(data[i:i + 2, 0], data[i:i + 2, 1],
                       c=data_start_color, alpha=above_tau_alpha)
            ax.plot(data[i:i + 2, 0], data[i:i + 2, 1],
                    c=data_start_color, alpha=above_tau_alpha)
        # above -> below tau
        elif data[i, 1] > tau and data[i + 1, 1] <= tau:
            ax.scatter(data[i, 0], data[i, 1],
                       c=data_start_color, alpha=above_tau_alpha)
            ax.scatter(data[i + 1, 0], data[i + 1, 1],
                       c=below_tau_color[i+1])
            ax.plot([data[i, 0], data[i, 0] + delta_x],
                    [data[i, 1], tau],
                    c=data_start_color, alpha=above_tau_alpha)
            # use the lower tau index color
            ax.plot([data[i, 0] + delta_x, data[i + 1, 0]],
                    [tau, data[i + 1, 1]],
                    c=below_tau_color[i+1])
        # below -> above tau
        elif data[i, 1] <= tau and data[i + 1, 1] > tau:
            ax.scatter(data[i, 0], data[i, 1], c=below_tau_color[i])
            ax.scatter(data[i + 1, 0], data[i + 1, 1],
                       c=data_start_color, alpha=above_tau_alpha)
            # use the lower tau index color
            ax.plot([data[i, 0], data[i, 0] + delta_x],
                    [data[i, 1], tau],
                    c=below_tau_color[i])
            ax.plot([data[i, 0] + delta_x, data[i + 1, 0]],
                    [tau, data[i + 1, 1]],
                    c=data_start_color, alpha=above_tau_alpha)
        # both below tau
        else:
            ax.scatter(data[i:i + 2, 0], data[i:i + 2, 1], c=below_tau_color[i])
            # use the lower tau index color
            if data[i, 1] <= data[i+1, 1]:
                ax.plot(data[i:i + 2, 0], data[i:i + 2, 1],
                        c=below_tau_color[i])
            else:
                ax.plot(data[i:i + 2, 0], data[i:i + 2, 1],
                        c=below_tau_color[i+1])

    ax.axhline(y=tau, c=tau_color, ls='--', lw=1.5, alpha=1, label='$\\tau$')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1))
    ax.set_title(title)

    if save:
        plt.savefig(os.path.join(path, name), dpi=dpi, bbox_inches='tight')
        plt.close()
    if show:
        plt.show()
        plt.close()

    return fig, ax
