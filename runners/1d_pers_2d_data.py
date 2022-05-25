"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
April, 2019
1d_pers_2d_data.py

script to build 1d persistence gif of 2d blowup disks to show loop
"""

import datetime
import dateutil.tz
from glob import glob
import matplotlib
matplotlib.use('agg')
import multidim
import numpy as np
import os
from sklearn.datasets import make_circles
from viz import holoviews_viz
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "1d_pers_2d_data_{}".format(datetime_stamp))
os.mkdir(out_path)

# how many points in circle
num_points = 150

# how noisy is circle
noise = 0.2

# total number of frames to go in the gif
num_frames =40

### building out our scenario ####

# create toy data
np.random.seed(27705)

data = make_circles(n_samples=num_points, noise=noise,
                   factor=0.999)[0] # sklearn function meant to be 2 circles, 0.999 makes ~1 circle

# build simplicial complex and run 0d persistence
pc = multidim.PointCloud(data, max_length=-1)
pc.make_pers1_rca1()

# persistence information
pers = pc.pers1.diagram

# let's think of persistence where the circles meet, not where one disk consumes a point
pers_half = pers.copy()
pers_half.loc[:, 'pers'] = pers_half.pers.values * 1/2
pers_half.loc[:, 'death'] = pers_half.death.values * 1/2
pers_half.loc[:, 'birth'] = pers_half.birth.values * 1/2

max_radius = pers_half.loc[:, 'death'].values.max()

# store holoviews figures to build holomap and save html widget
figures = []

# we will break up gif into before birth of big circle and after
birth_big_circle = pers_half.sort_values(by='death', ascending=False).loc[:, 'birth'].values[0]

# before loop forms
for i, radius in enumerate(np.linspace(0, birth_big_circle, num_frames // 3)):
    fig, ax1, ax2 = viz.pers_0d_disk_viz(data, pers_half, radius, buffer=2,
                                         path=out_path, name='pers_1d_pre{:03}.png'.format(i),
                                         dpi=100, save=True, show=False)

    # make holoviews figures
    figures.append((holoviews_viz.viz_disks_hv(data, radius, padding=2 / 40) +
                    holoviews_viz.viz_persistence_hv(pers_half, radius, 2 / 40)).\
                        opts(fig_inches=(12, 6), sublabel_format=""))

# after loop forms
for i, radius in enumerate(np.linspace(birth_big_circle, max_radius + 0.01, 2 * num_frames // 3)):
    fig, ax1, ax2 = viz.pers_0d_disk_viz(data, pers_half, radius, buffer=2,
                                         path=out_path, name='pers_1d_post{:03}.png'.format(i),
                                         dpi=100, save=True, show=False)

    # make holoviews figures
    figures.append((holoviews_viz.viz_disks_hv(data, radius, padding=2 / 40) +
                    holoviews_viz.viz_persistence_hv(pers_half, radius, 2 / 40)).\
                        opts(fig_inches=(12, 6), sublabel_format=""))

#### make dynamic viz ####

## holoviews html widget ##

hv_radius_values = np.concatenate((np.linspace(0, birth_big_circle, num_frames // 3),
                                   np.linspace(birth_big_circle, max_radius + 0.01, 2 * num_frames // 3)))

holomap = holoviews_viz.make_hv_widget_one_variable(figures, hv_radius_values,
                                                    "Radius", save=True,
                                                    path=out_path, file_name='1d_pers_2d_data_widget.html')

## gif ##

# pause a little more than normal between frames, and 3 second pause at end
os.system('convert -delay 30> -loop 0 {} -delay 300 {} -delay 30 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'pers_1d_pre*'),
    sorted(glob(os.path.join(out_path, 'pers_1d_pre*')))[-1],
    os.path.join(out_path, 'pers_1d_post*'),
    sorted(glob(os.path.join(out_path, 'pers_1d_post*')))[-1],
    os.path.join(out_path, '1d_disks')))