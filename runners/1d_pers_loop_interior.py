"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
1d_pers_loop_interior.py

script to build 1d persistence figures with two different exterior shapes of a loop
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
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "1d_pers_loop_interior_{}".format(datetime_stamp))
os.mkdir(out_path)

# how many points in circle
num_points = 100

# how noisy is circle
noise = 0

# points per dimension of exterior grid
num_grid_pts = 20

### building out our scenario ####

# create toy data
np.random.seed(27705)

data = make_circles(n_samples=num_points, noise=noise,
                   factor=0.999)[0] # sklearn function meant to be 2 circles, 0.999 makes ~1 circle

# make grid
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, num_grid_pts),
                   np.linspace(-1.5, 1.5, num_grid_pts))
grid = np.c_[X.flatten(), Y.flatten()]

# only keep grid points outside the circle
to_keep = grid[np.where((grid**2).sum(axis=1) >= 1)[0], :]

for dataset, name in zip([data, np.r_[data, to_keep]], ['circle', 'circle_with_grid']):

    # build simplicial complex and run 0d persistence
    pc = multidim.PointCloud(dataset, max_length=-1)
    pc.make_pers1_rca1()

    # persistence information
    pers = pc.pers1.diagram

    # let's think of persistence where the circles meet, not where one disk consumes a point
    pers_half = pers.copy()
    pers_half.loc[:, 'pers'] = pers_half.pers.values * 1/2
    pers_half.loc[:, 'death'] = pers_half.death.values * 1/2
    pers_half.loc[:, 'birth'] = pers_half.birth.values * 1/2

    max_radius = pers_half.loc[:, 'death'].values.max()

    viz.pers_with_data_viz(dataset, pers_half,
                           save=True, path=out_path, name='pers_1d_{}.png'.format(name))
