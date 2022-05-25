"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019

3d_surface_scenario.py

make gif of scenario on which we will run lowerstar filtration
"""

import datetime
import dateutil.tz
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "3d_scenario_{}".format(datetime_stamp))
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
fig, ax = plt.subplots(figsize=(6, 6))
a = ax.imshow(density, origin='low')
ax.axis('off')
ax.set_title('3d Scenario')
cb = fig.colorbar(a)
cb.ax.set_title("Height")
plt.savefig(os.path.join(out_path, 'density_viz_static.png'))

# keep track of image number to save accordingly
index = 0

# start from above, get to `above_angle` (desired height above XY plane)
to_desired_height = [i for i in reversed(np.arange(above_angle, 91, above_step))]
for z_rotation in to_desired_height:
    viz.plot_3d_surface(X, Y, density, z_rotation, start_angle,
                        save=True, path=out_path,
                        name='3d_scenario_{:03}.png'.format(index),
                        dpi=100)
    index += 1

# rotate the axes on maintained desired height above XY plane
for angle in range(start_angle, start_angle + 360 + z_step, z_step):
    viz.plot_3d_surface(X, Y, density, to_desired_height[-1], angle,
                        save=True, path=out_path,
                        name='3d_scenario_{:03}.png'.format(index),
                        dpi=100)
    index +=1

# get back to birds eye view
for z_rotation in reversed(to_desired_height):
    viz.plot_3d_surface(X, Y, density, z_rotation, start_angle,
                        save=True, path=out_path,
                        name='3d_scenario_{:03}.png'.format(index),
                        dpi=100)
    index += 1

os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, '3d_scenario_*'),
    sorted(glob(os.path.join(out_path, '3d_scenario_*')))[-1],
    os.path.join(out_path, '3d_scenario')))