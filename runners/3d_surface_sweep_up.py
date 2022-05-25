"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019

3d_surface_sweep_up.py

make gif of sweeping threshold up on scenario on which we will run lowerstar filtration
shown in both image form and rotating 3d form
"""

import datetime
import dateutil.tz
from glob import glob
import numpy as np
import os
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "3d_scenario_sweep_{}".format(datetime_stamp))
os.mkdir(out_path)

# angle above the XY plane to go in circle
above_angle = 70

# starting angle around Z axis
start_angle = -90

# step from overhead to above_angle and back
above_step = 4

# step around Z axis
z_step = 3

# dataset
data = np.load('./for_lowerstar.npz')
density = data['density']
X = data['X']
Y = data['Y']

#### build gif ####

# static scenario
viz.plot_image(density, path=out_path, name='density_viz_static.png')

# keep track of image number to save accordingly
index = 0

# find max color on density
vmin = 0
vmax = density.max()

# rotation around z axis
rotation = [i for i in range(start_angle, start_angle + 360 + z_step, z_step)]

# sweeping up threshold at same time
threshold = np.linspace(vmin, vmax, len(rotation))

# rotate the axes on maintained desired height above XY plane while sweeping up threshold
for angle, thresh in zip(rotation, threshold):
    # cmap = viz.categorical_cmap(threshold=thresh, base_cmap='viridis',
    #                             vmin=vmin, vmax=vmax, below_color='C3')
    cmap = viz.categorical_cmap(threshold=thresh, base_cmap='viridis',
                                vmin=vmin, vmax=vmax, above_color='white')

    # 3d rotation viz
    viz.plot_3d_surface(X, Y, density, above_angle, angle, cmap=cmap,
                        save=True, path=out_path,
                        name='3d_scenario_sweep_{:03}.png'.format(index),
                        dpi=100)

    # image viz with same cmap
    viz.plot_image(density, cmap=cmap,
                   save=True, path=out_path,
                   name='image_scenario_sweep_{:03}.png'.format(index),
                   dpi=100)

    index +=1

# 3d rotation gif
os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, '3d_scenario_sweep_*'),
    sorted(glob(os.path.join(out_path, '3d_scenario_sweep_*')))[-1],
    os.path.join(out_path, '3d_scenario_sweep')))

# image gif
os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'image_scenario_sweep_*'),
    sorted(glob(os.path.join(out_path, 'image_scenario_sweep_*')))[-1],
    os.path.join(out_path, 'image_scenario_sweep')))