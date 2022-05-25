"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
April, 2019
0d_pers_2d_data.py

script to build 0d persistence gif of 2d persistence blowup
"""

import datetime
import dateutil.tz
from glob import glob
from homology.dim0 import all_roots
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multidim
import numpy as np
import os
from viz import holoviews_viz
from viz import viz

#### hardcoded info ####

# whether to build single color below threshold or show connected components as different colors
show_components = True

# what cmap to use
if show_components:
    # same colored disks as other examples without color
    cmap = plt.cm.Paired
else:
    # makes better colors for multi-colored disks
    cmap = plt.cm.tab10

# buffer around points to avoid resizing as disks grow
buffer = 5

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "0d_pers_2d_data_{}".format(datetime_stamp))
os.mkdir(out_path)

# two centers, how many points in each center
pts_per_center = 5

center_0 = 3
center_1 = 9

sd_0 = 0.7
sd_1 = 1

# total number of frames to go in the gif
num_frames = 50

### building out our scenario ####

# create toy data
np.random.seed(27701)
dat_0 = np.random.normal(center_0, sd_0, pts_per_center*2).reshape(-1, 2)
dat_1 = np.random.normal(center_1, sd_1, pts_per_center*2).reshape(-1, 2)

data = np.r_[dat_0, dat_1]

# build simplicial complex and run 0d persistence
pc = multidim.PointCloud(data, max_length=-1)
pc.make_pers0()

# persistence information
# first point is birth index 0, de)ath index 0, the representation of the infinity pers point
#  (dropping it for interpretation purposes
pers = pc.pers0.diagram.iloc[1:, :]

# let's think of persistence where the circles meet, not where one disk consumes a point
pers_half = pers.copy()
pers_half.loc[:, 'pers'] = pers_half.pers.values * 1/2
pers_half.loc[:, 'death'] = pers_half.death.values * 1/2

max_radius = pers_half.loc[:, 'pers'].values.max()

# will break up gif into before 2 clusters and after
birth_2_clusters = pers_half.sort_values(by='death', ascending=False).loc[:, 'death'].values[1]

# store holoviews figures to build holomap and save html widget
figures = []

# before 2 clusters
for i, radius in enumerate(np.linspace(0, birth_2_clusters, num_frames//2)):

    # get components at the cutoff we're running for colors
    if show_components:
        pc.reset()
        # replicate meeting at halfway point by running pers diagram twice as far
        pc.make_pers0(cutoff=2*radius + 1e-15)
        roots = pc.stratum[0]['rep'].values.copy()
        all_roots(roots)
        colors = np.array(roots)
        # max color will be highest index of a root e.g. num_roots - 1
        vmax = data.shape[0] - 1
    else:
        colors = 0
        vmax = 1

    fig, ax1, ax2 = viz.pers_0d_disk_viz(data, pers_half, radius, disk_colors=colors,
                                         cmap=cmap, vmin=0, vmax=vmax,
                                         buffer=buffer,
                                         path=out_path, name='pers_0d_pre_{:03}.png'.format(i),
                                         dpi=100, save=True)

    # make holoviews figures
    figures.append((holoviews_viz.viz_disks_hv(data, radius, padding=buffer/40,
                                               disk_colors=colors, cmap=cmap) +
                    holoviews_viz.viz_persistence_hv(pers_half, radius, buffer/40)).\
                        opts(fig_inches=(12, 6), sublabel_format=""))

# after 2 clusters
for i, radius in enumerate(np.linspace(birth_2_clusters, max_radius + 0.01, num_frames//2)):

    # get components at the cutoff we're running for colors
    if show_components:
        pc.reset()
        # replicate meeting at halfway point by running pers diagram twice as far
        pc.make_pers0(cutoff=2*radius + 1e-15)
        roots = pc.stratum[0]['rep'].values.copy()
        all_roots(roots)
        colors = np.array(roots)
        # max color will be highest index of a root e.g. num_roots - 1
        vmax = data.shape[0] - 1
    else:
        colors = 0
        vmax=1

    fig, ax1, ax2 = viz.pers_0d_disk_viz(data, pers_half, radius, disk_colors=colors,
                                         cmap=cmap, vmin=0, vmax=vmax,
                                         buffer=buffer,
                                         path=out_path, name='pers_0d_post_{:03}.png'.format(i),
                                         dpi=100, save=True)

    # make holoviews figures
    figures.append((holoviews_viz.viz_disks_hv(data, radius, padding=buffer / 40,
                                               disk_colors=colors, cmap=cmap) +
                    holoviews_viz.viz_persistence_hv(pers_half, radius, buffer / 40)).\
                   opts(fig_inches=(12, 6), sublabel_format=""))

#### make dynamic viz ####

## holoviews html widget ##

hv_radius_values = np.concatenate((np.linspace(0, birth_2_clusters, num_frames//2),
                                   np.linspace(birth_2_clusters, max_radius + 0.01, num_frames//2)))

holomap = holoviews_viz.make_hv_widget_one_variable(figures, hv_radius_values,
                                                    "Radius", save=True,
                                                    path=out_path, file_name='0d_pers_2d_data_widget.html')

## panel html widget ##

# for holoviews-based widget, need to precompute the bounds of the disks graph
#  (otherwise the bounds aren't fixed)

def viz_function(radius):
    """
    one param viz function to be called in making the panel-based html widget

    :param radius:
    :return:
    """

    #### some global variables we need to modify for each radius ####

    pc.reset()
    # replicate meeting at halfway point by running pers diagram twice as far
    pc.make_pers0(cutoff=2 * radius + 1e-15)
    roots = pc.stratum[0]['rep'].values.copy()
    all_roots(roots)
    colors = np.array(roots)
    # max color will be highest index of a root e.g. num_roots - 1
    vmax = data.shape[0] - 1

    # return the actual viz

    # using matplotlib figures
    # fig, ax0, ax1 = viz.pers_0d_disk_viz(data, pers_half, radius, disk_colors=colors,
    #                                      cmap=cmap, vmin=0, vmax=vmax,
    #                                      buffer=buffer, show=False,
    #                                      dpi=100, save=False)
    # plt.close(fig)
    # return fig

    # using holoviews figures
    #### BUGGY STILL, use with caution  #####
    return \
        (holoviews_viz.viz_disks_hv(data, radius, padding=buffer / 40,
                                    disk_colors=colors, cmap=cmap, vmin=0, vmax=vmax) +
         holoviews_viz.viz_persistence_hv(pers_half, radius, buffer / 40)). \
            opts(fig_inches=(12, 6), sublabel_format="")


test_app = holoviews_viz.make_panel_widget_one_variable(viz_function, hv_radius_values, "Radius",
                                                        title='0d Persistent Homology',
                                                        save=True, path=out_path,
                                                        file_name='0d_pers_2d_data_panel_widget.html')

## gif ##

gif_name = '0d_disks'
if show_components:
    gif_name += '_components'
# pause a little more than normal between frames, and 3 second pause at end
os.system('convert -delay 30> -loop 0 {} -delay 300 {} -delay 30 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'pers_0d_pre*'),
    sorted(glob(os.path.join(out_path, 'pers_0d_pre*')))[-1],
    os.path.join(out_path, 'pers_0d_post*'),
    sorted(glob(os.path.join(out_path, 'pers_0d_post*')))[-1],
    os.path.join(out_path, gif_name)))