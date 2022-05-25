"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019

lowerstar_sweep_up.py

make gif of sweeping threshold up on scenario with lowerstar filtration
shown in both image form and rotating 3d form
"""

import datetime
import dateutil.tz
from glob import glob
from homology.dim0 import all_roots
import multidim
import numpy as np
import numpy.ma as ma
import os
from viz import viz

#### hardcoded info ####

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "lowerstar_sweep_{}".format(datetime_stamp))
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
vmax = density.max() + 0.01

# rotation around z axis
rotation = [i for i in range(start_angle, start_angle + 360 + z_step, z_step)]

# sweeping up threshold at same time
threshold = np.linspace(vmin, vmax, len(rotation))

# run lower star filtration on image
lowerstar = multidim.lower_star_for_image(density,
                                          diagonals_and_faces=False)

# use same random cmap throughout for image
np.random.seed(27701)
# need at most a random color index for every pixel of `density`
image_cmap = viz.random_cmap(np.arange(density.size))

# rotate the axes on maintained desired height above XY plane while sweeping up threshold
for angle, thresh in zip(rotation, threshold):
    # reset lowerstar to build up connections to exact threshold of interest
    lowerstar.reset()
    # build simplicial complex up to appropriate connectedness
    lowerstar.make_pers0(cutoff=thresh, show_diagonal=True)
    # pull out information of how indices are connected
    roots = lowerstar.stratum[0]['rep'].values.copy()
    all_roots(roots)
    roots.shape = density.shape
    # mask values not yet swept up to
    mask = ma.masked_where(density > thresh, roots)

    # use same random cmap throughout (but white out root indices we haven't reached yet)
    # np.random.seed(27701)
    # root_indices_to_ignore = np.where(ma.getmask(mask)!=True)
    # # need at most a random color index for every pixel of `density`
    # new_cmap = viz.random_cmap(np.arange(density.size), indices_to_change=root_indices_to_ignore)

    # 3d rotation viz
    # viz.plot_3d_surface(X, Y, density, above_angle, angle, cmap=new_cmap, include_colorbar=False,
    #                     save=True, path=out_path,
    #                     name='3d_lowerstar_sweep_{:03}.png'.format(index),
    #                     dpi=100)

    # image viz with same cmap
    viz.plot_image(mask, cmap=image_cmap, include_colorbar=False,
                   title='Connected Components at Height {:.02f}'.format(thresh),
                   save=True, path=out_path,
                   name='image_lowerstar_sweep_{:03}.png'.format(index),
                   dpi=100)

    index +=1

# 3d rotation gif
# os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
#     os.path.join(out_path, '3d_lowerstar_sweep_*'),
#     sorted(glob(os.path.join(out_path, '3d_lowerstar_sweep_*')))[-1],
#     os.path.join(out_path, '3d_lowerstar_sweep')))

# image gif
os.system('convert -loop 0 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'image_lowerstar_sweep_*'),
    sorted(glob(os.path.join(out_path, 'image_lowerstar_sweep_*')))[-1],
    os.path.join(out_path, 'image_lowerstar_sweep')))