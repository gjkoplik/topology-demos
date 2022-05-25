"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
1d_pers_three_loops.py

script to build 1d persistence gif as data pulled to three loops
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
out_path = os.path.join('./figures', "1d_pers_three_loops_{}".format(datetime_stamp))
os.mkdir(out_path)

# data range
data_min = 0
data_max = 10

# we will do a meshgrid of data, this is number of points along each dimension
point_span = 15

# three loop centers
center_0 = np.array([8, 8])
center_1 = np.array([8, 2])
center_2 = np.array([3, 5])

# size of each center's radius
radii = np.array([1, 2, 2.5])

# max y value on output persistence diagrams (roughly radius of largest center)
pers_max = 2.5

# total data points
num_points = 300

# total number of frames to go in the gif
num_frames = 50

# number of vectors along each dimension of the vector field
num_vectors = 50

# extent to which points follow vector field in each step
update_scalar=0.1

### building out our scenario ####

# create toy data
np.random.seed(27705)

X, Y = np.meshgrid(np.linspace(0, 10, point_span),
                   np.linspace(0, 10, point_span))
data = np.c_[X.flatten(), Y.flatten()]


# build vector field
X, Y, U, V = mvf.vf_3_loops(num_x=num_vectors, num_y=num_vectors, radii=radii,
                                    center_0=center_0, center_1=center_1, center_2=center_2)

# save for record
viz.plot_vector_field(X, Y, U, V, save=True, path=out_path)

# run initial state viz
# build point cloud and run 0d persistence
pc = multidim.PointCloud(data, max_length=-1)
pc.make_pers1_rca1()
# output viz
viz.pers_with_data_viz(data, pc.pers1.diagram,
                       pers_max=pers_max,
                       data_xrange=(data_min - 1, data_max + 1), data_yrange=(data_min - 1, data_max + 1),
                       save=True, path=out_path, name='pers_1d_three_loop_{:03}.png'.format(0))

# run loop pulling the data closer and closer towards the three loops
temp_data = data.copy()
for i in range(1, num_frames):
    # interpolate vf for current data points
    interpolated_vf = mvf.interpolate_vector_field_to_data((X, Y, U, V), temp_data)
    # move data points according to interpolated vf
    temp_data = mvf.update_data_location(temp_data, interpolated_vf, update_scalar=update_scalar)
    # build point cloud and run 0d persistence
    pc = multidim.PointCloud(temp_data, max_length=-1)
    pc.make_pers1_rca1()
    # let's think of persistence where the circles meet, not where one disk consumes a point
    pers = pc.pers1.diagram
    pers_half = pers.copy()
    pers_half.loc[:, 'pers'] = pers_half.pers.values * 1 / 2
    pers_half.loc[:, 'death'] = pers_half.death.values * 1 / 2
    pers_half.loc[:, 'birth'] = pers_half.birth.values * 1 / 2
    # output viz
    viz.pers_with_data_viz(temp_data, pers_half,
                           pers_max=pers_max,
                           data_xrange=(data_min - 1, data_max + 1), data_yrange=(data_min - 1, data_max + 1),
                           save=True, path=out_path, name='pers_1d_three_loop_{:03}.png'.format(i))

# 3 second pause at end of gif
os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'pers_1d_three_loop_*'),
    sorted(glob(os.path.join(out_path, 'pers_1d_three_loop_*')))[-1],
    os.path.join(out_path, 'pers_1d_three_loop')))