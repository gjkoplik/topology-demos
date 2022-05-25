"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
July, 2019
signal_compression.py

An example showing how persistent signal compression preserves structure using a random walk
"""

from copy import copy
import datetime
import dateutil.tz
from glob import glob
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from timeseries import Signal
from viz import holoviews_viz
from viz import viz

#### setup ####

# length of signal
signal_length = 1000

# max cutoff to threshold persistence diagram for reconstruction
persistence_cutoff = 10

# plot removing every `skip` persistence values
skip = 5

# two different positive / negative values
choices = [-1, -0.5, 0.5, 1]

# subset of indices to look at to observe lost information when compressing with persistence
sub_min = 150
sub_max = 200

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "signal_compression_{}".format(datetime_stamp))
os.mkdir(out_path)

# where the frames of the gif will be stored
image_path = os.path.join(out_path, 'images')
os.mkdir(image_path)

hv_image_path = os.path.join(out_path, "hv_images")
os.mkdir(hv_image_path)

#### reconstruction function based on persistence diagram ####

def signal_persistence(signal: np.ndarray):
    """
    Build a persistence diagram for a signal

    Parameters
    ----------
    signal : (n, 2) array where the first column represents the times / domain / x points of the signal
        and the second column represents the scalars / range / y points corresponding to each time.

    Return
    ------
    pandas.DataFrame of signal persistence
    """
    ts = Signal(values=signal[:, 1], times=signal[:, 0])
    ts.make_pers()
    return ts.pers.diagram


def reconstruct_signal(signal: np.ndarray, pers_diag: pd.DataFrame = None, persistence_cutoff: int = None,
                       num_indices_to_keep: int or str = None):
    """
    Removes the least persistent critical point pairs from a signal on a 1-dimensional domain and
    reconstructs the signal according to a linear approximation in between dropped critical point pairs.

    To maintain spanning the same domain of the signal, if the edge values are not top critical points,
    (or not critical points at all), we will still return the signal values there. When specifying
    `num_indices_to_keep = n`, the function will still return exactly `n` values, subtracting index values
    if needing to add in signal values. When specifying `persistence_cutoff = k`, the function will return
    all critical points with persistence greater than `k` along with the edge values if not already included.

    An empty persistence diagram or choosing to keep zero indices from the persistence diagram will result in
    a reconstruction of a constant line spanning the domain valued at the median of the signal.

    Parameters
    ----------
    signal : (n, 2) array where the first column represents the times / domain / x points of the signal
        and the second column represents the scalars / range / y points corresponding to each time.

    pers_diag : output of persistence information as called from `signal_persistence()`.
        Default `None` computes persistence diagram for `signal` on the fly.

    persistence_cutoff : a non-`None` value will reconstruct the signal only using critical points pairs with
        persistence greater than this value.
    Note: the user can either specify a `persistence_cutoff`, or retain a desired number of points using
        `num_indices_to_keep, but exactly one of these must be specified as non-`None`.

    num_indices_to_keep : number of points to keep from signal
        when reconstructing the signal. Can also offer the string "all" here for a reconstruction using the full
        persistence diagram. Must specify an integer greater than 2 (will always return at least the edge values)
    Note: the user can either specify a `persistence_cutoff`, or retain a desired number of points using
        `num_indices_to_keep, but exactly one of these must be specified as non-`None`.

    :return (k, 2) np.ndarray (k <=n) representing the reconstructed signal where the first column represents the
        times / domain / x points of the signal and the second column represents the scalars / range / y points
        corresponding to each time.

    Notes
    -----
    The Morse Cancellation Lemma guarantees that the dot of lowest persistence corresponds to
    a pair of *adjacent* critical points (and also guarantees that "un-kinking" that pair of critical
    points will not un-kink any other pair).

    Technically, this Lemma requires that the two neighbors of any given critical point be distinct.
    If this assumption fails, we could create non-unique solutions. However, leaving this at the mercy
    of pandas sorting by persistence will still return a *correct* result regardless, even though it may
    not be unique.

    Furthermore, this Lemma only applies to removing the lowest persistence critical point, but as long as we
    remove the n lowest critical points, the order in which we remove them should not matter (in other words, at
    least when replacing critical points with linear intepolation, removing these critical points is commutative).


    References
    ----------
    Edelsbrunner, Herbert, Dmitriy Morozov, and Valerio Pascucci.
    "Persistence-sensitive simplification functions on 2-manifolds."
    Proceedings of the twenty-second annual symposium on Computational geometry. 2006.
    """

    if persistence_cutoff is None and num_indices_to_keep is None:
        raise NotImplementedError(
            "Must Specify either `persistence_cutoff` or `num_indices_to_keep`"
        )

    if persistence_cutoff is not None and num_indices_to_keep is not None:
        raise NotImplementedError(
            "Must Specify *exactly* one of `persistence_cutoff` or `num_indices_to_keep`"
        )

    if num_indices_to_keep is not None and num_indices_to_keep != "all":
        assert num_indices_to_keep >= 2, \
            "`num_indices_to_keep` must be >=2 since we always at least return the edge values"

    # we will reference the Signal object for knowing min and max indices later
    ts = Signal(values=signal[:, 1], times=signal[:, 0])

    if pers_diag is None:
        ts.make_pers()
        pers_diag = ts.pers.diagram

    # work with a sorted persistence diagram to remove the lowest persistence points
    pers_diag = pers_diag.sort_values('pers', ascending=False)

    # know the index bounds so we can guarantee we maintain the domain
    start_index = ts.components.index.min()
    stop_index = ts.components.index.max()

    # take our desired subset of the persistence diagram for reconstruction
    if num_indices_to_keep is not None:

        if num_indices_to_keep == "all":
            # all critical pairs + endpoints
            indices = [start_index, stop_index] + \
                      list(pers_diag.loc[:, ['birth_index', 'death_index']].values.flatten())
            # remove redundancies
            indices = list(set(indices))

        else:
            # exactly `num_indices_to_keep` points
            # start with all possible values
            ordered_indices = [start_index, stop_index] + \
                              list(pers_diag.loc[:, ['birth_index', 'death_index']].values.flatten())

            # if we're already below our threshold for number of values, stop
            if len(list(set(ordered_indices))) <= num_indices_to_keep:
                indices = list(set(ordered_indices))

            # otherwise make sure we hit our exact cutoff
            else:
                # make sure we get the top *unique* indices:
                num_indices = num_indices_to_keep
                while True:
                    new_indices = ordered_indices[:num_indices]
                    if len(list(set(new_indices))) == num_indices_to_keep:
                        break
                    num_indices += 1
                # remove redundancies
                indices = list(set(new_indices))

    elif persistence_cutoff is not None:
        pers_diag = pers_diag[pers_diag.pers > persistence_cutoff]

        indices = [start_index, stop_index] + \
                  list(pers_diag.loc[:, ['birth_index', 'death_index']].values.flatten())

        # remove redundancies
        indices = list(set(indices))

    else:
        raise NotImplementedError

    return signal[sorted(indices), :]

#### building and compressing signal ####

np.random.seed(27703)
# random walk plus a little bit of noise to separate values on pers diagram
data = np.cumsum(np.random.choice(choices, signal_length)) + np.random.normal(0, 0.1, signal_length)
times = np.arange(data.size)

# plot original figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times, data)
ax.set_title(f'Original Signal ({signal_length} Data Points)')
plt.savefig(os.path.join(out_path, "original_signal.png"),
            bbox_inches='tight', dpi=200)
plt.close()

ts = Signal(values=data, times=times)
ts.make_pers()
pers = ts.pers.diagram.sort_values('pers')
viz.plot_persistence(pers, title='Signal Persistence',
                     show=False, save=True,
                     path=out_path, name='original_signal_pers_diagram.png')

# plot compressed figure with only persistence diagram
compressed_signal = reconstruct_signal(signal=np.c_[times, data], pers_diag=pers,
                                       num_indices_to_keep="all")

# scale up compressed reconstruction to same shape as original signal
V = pd.Series(index=np.arange(data.size), dtype=np.float64)
V[compressed_signal[:, 0]] = compressed_signal[:, 1]
# fill in the missing values with linear interpolation
V.interpolate(method='linear', limit_area="inside", inplace=True)
# overwrite compressed signal to resized version
compressed_signal = np.c_[V.index, V.values]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(compressed_signal[:, 0], compressed_signal[:, 1], c='C1')
ax.set_title(f'Signal Reconstructed from Persistence Diagram ({pers.shape[0]} Data Points)')
plt.savefig(os.path.join(out_path, "full_pers_signal.png"),
            bbox_inches='tight', dpi=200)
plt.close()

# plot the compressed with the original
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times, data, label='Original Signal')
ax.plot(compressed_signal[:, 0], compressed_signal[:, 1], '--',
        label='Reconstructed Signal')
ax.set_title('Reconstructed Signal "Holds True" to Original Signal')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(out_path, "original_with_compressed_signal.png"),
            bbox_inches='tight', dpi=200)
plt.close()

# plot subset to demonstrate only capturing critical points (missing changes that aren't critical points)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times[sub_min:sub_max], data[sub_min:sub_max],
        lw=2, label='Original Signal')
ax.plot(compressed_signal[sub_min:sub_max, 0],
        compressed_signal[sub_min:sub_max, 1],
        '--', label='Reconstructed Signal',
        lw=2, alpha=0.8)
ax.set_title(f'Some Information Lost When Reconstructing Signal with Persistence Diagram')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(os.path.join(out_path, "original_with_compressed_signal_subset.png"),
            bbox_inches='tight', dpi=200)
plt.close()

# store holoviews figures to build holomap and save html widget
figures = []

# reference original persistence diagram
original_pers = copy(pers)

# loop over removing  every `skip` pers values and reconstructing signal (and check with just 2 values e.g. endpoints)
num_pers_values_removed = np.arange(0, pers.shape[0], skip)
for i, val in enumerate(num_pers_values_removed):
    # threshold the persistence diagram

    # build compressed signal via subset of original persistence diagram
    compressed_signal = reconstruct_signal(signal=np.c_[times, data], pers_diag=pers,
                                           num_indices_to_keep=pers.shape[0]-val)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4),
                                   gridspec_kw={'width_ratios': [10., 4.]})
    ax0.plot(compressed_signal[:, 0], compressed_signal[:, 1])

    # don't need number of points for holoviews title
    title = f'Signal Reconstructed from Partial Persistence Diagram'

    # grammar handling for number of points for matplotlib title
    mpl_title = title + f' ({pers.shape[0]-val} Data Point'
    if pers.shape[0]-val == 1:
        mpl_title += ')'
    else:
        mpl_title += 's)'

    ax0.set_title(mpl_title)

    # corresponding pers diagram
    # work with a sorted persistence diagram to remove the lowest persistence points
    pers_diag = pers.copy().sort_values('pers', ascending=False)

    # know the index bounds so we can guarantee we maintain the domain
    start_index = ts.components.index.min()
    stop_index = ts.components.index.max()

    # take our desired subset of the persistence diagram for reconstruction
    pers_diag = pers_diag.iloc[:pers.shape[0]-val, :]

    # make sure we also have the edges of the signal
    indices = list(pers_diag.loc[:, ['birth_index', 'death_index']].values.flatten())
    start_edge_contained = start_index in indices
    stop_edge_contained = stop_index in indices

    # but still the total desired number of indices
    num_vals_to_drop = int(np.logical_not(start_edge_contained)) + int(np.logical_not(stop_edge_contained))
    # remove the values from the original persistence diagram
    pers_diag_remaining = pers.sort_values('pers', ascending=False).iloc[:pers.shape[0]-val - num_vals_to_drop, :]
    pers_diag_removed = pers.sort_values('pers', ascending=False).iloc[(pers.shape[0]-val-num_vals_to_drop): , :]

    viz.plot_persistence(pers_diag_remaining, title='Signal Persistence',
                         show=False, save=False,
                         fig=fig, ax=ax1,
                         color='C0')
    ax1.scatter(pers_diag_removed.loc[:, 'birth'].values,
                pers_diag_removed.loc[:, 'death'].values,
                c='C1', alpha=0.5)

    # add legend
    remaining = plt.scatter([], [], c='C0',
                            alpha=0.5)
    removed = plt.scatter([], [], c='C1',
                          alpha=0.5)
    ax1.legend([remaining, removed],
               ['Remaining\nPersistence Values', 'Removed\nPersistence Values'],
               loc='upper left', bbox_to_anchor=(1,1))

    plt.savefig(os.path.join(image_path, "signal_{:03}.png".format(i)), bbox_inches='tight')
    plt.close()

    ## make holoviews figures ##

    signal = \
        holoviews_viz.viz_signal_hv(compressed_signal[:, 0], compressed_signal[:, 1],
                                    aspect=2.5, padding=2 / 40,
                                    title=title)

    # plot persistence values kept and removed
    kept_points = \
        holoviews_viz.viz_persistence_hv(pers_diag_remaining, None,
                                         label='Remaining Values')

    removed_points = \
        hv.Points(pers_diag_removed.loc[:, ['birth', 'death']].values,
                  label='Removed Values'). \
            opts(color='orange', s=50, alpha=0.7)

    # only show legend if we have some removed points
    # show_legend = True
    pers_graph = (kept_points * removed_points).opts(show_legend=True,
                                                     legend_position='right')

    # put into single figure
    hv_figure = \
        (signal + pers_graph). \
            opts(aspect_weight=True,
                 fig_inches=10,
                 tight=True,
                 sublabel_format="")

    figures.append(hv_figure)
    hv.save(hv_figure, os.path.join(hv_image_path, "signal_{:03}.png".format(i)), fmt='png')

#### build dynamic viz ####

## holoviews html widget ##

# Number of remaining pers values is the total number minus the number of values we are skipping
holomap = holoviews_viz.make_hv_widget_one_variable(figures, num_pers_values_removed,
                                                    "Number of Removed Values", save=True,
                                                    path=out_path, file_name='signal_compression_widget.html')

## make gif ##

os.system('convert -delay 30> -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(hv_image_path, 'signal_*'),
    sorted(glob(os.path.join(hv_image_path, 'signal_*')))[-1],
    os.path.join(out_path, 'compressed_signal')
  )
)