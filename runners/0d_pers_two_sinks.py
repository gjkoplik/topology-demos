"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
0d_pers_two_sinks.py

script to build 0d persistence gif as data pulled to two sinks
"""

import datetime
import dateutil.tz
from glob import glob
import matplotlib
matplotlib.use('agg')
import multidim
import numpy as np
import os
from scipy.spatial.distance import euclidean
from vector_fields import make_vector_fields as mvf
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "0d_pers_two_sinks_{}".format(datetime_stamp))
os.mkdir(out_path)

# data range
data_min = 0
data_max = 10

# two centers
center_0 = np.array([2, 7])
center_1 = np.array([7, 3])

# max y value on output persistence diagrams
#   defaulting to euclidean distance between centers / 2 + 1
pers_max = euclidean(center_0, center_1) / 2 + 1

# total data points
num_points = 300

# total number of frames to go in the gif
num_frames = 40

# extent to which points follow vector field in each step
update_scalar=0.2

### building out our scenario ####

# create toy data
np.random.seed(27705)

data = np.random.uniform(data_min, data_max, num_points*2).reshape(-1, 2)

# build vector field
X, Y, U, V, c0, c1 = mvf.vf_2_sinks(center_0=center_0, center_1=center_1)

# save for record
viz.plot_vector_field(X, Y, U, V, save=True, path=out_path)

# run loop pulling the data closer and closer towards the two sinks
temp_data = data.copy()
for i in range(num_frames):
    # interpolate vf for current data points
    interpolated_vf = mvf.interpolate_vector_field_to_data((X, Y, U, V), temp_data)
    # move data points according to interpolated vf
    temp_data = mvf.update_data_location(temp_data, interpolated_vf, update_scalar=update_scalar)
    # build point cloud and run 0d persistence
    pc = multidim.PointCloud(temp_data, max_length=-1)
    pc.make_pers0()

    # let's think of persistence where the circles meet, not where one disk consumes a point
    pers = pc.pers0.diagram
    pers_half = pers.copy()
    pers_half.loc[:, 'pers'] = pers_half.pers.values * 1 / 2
    pers_half.loc[:, 'death'] = pers_half.death.values * 1 / 2

    # output viz
    viz.pers_with_data_viz(temp_data, pers_half,
                           pers_max=pers_max,
                           data_xrange=(data_min - 1, data_max + 1), data_yrange=(data_min - 1, data_max + 1),
                           save=True, path=out_path, name='pers_0d_two_sink_{:03}.png'.format(i))

# 3 second pause at end of gif
os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'pers_0d_two_sink_*'),
    sorted(glob(os.path.join(out_path, 'pers_0d_two_sink_*')))[-1],
    os.path.join(out_path, '0d_two_sink')))