"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
May, 2019
signal_sweep.py

script to build gif of a toy signal, sweeping threshold up and
    observing connected components plus persistence diagram
"""

import datetime
import dateutil.tz
from glob import glob
from homology.dim0 import all_roots
import matplotlib.pyplot as plt
import multidim
import numpy as np
import os
from timeseries import Signal
from viz import viz

#### hardcoded info ####

# whether to build single color below threshold or show connected components as different colors
show_components = True

# what the single color should be if `show_components = False`
single_color = 'C0'

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "signal_sweep_{}".format(datetime_stamp))
os.mkdir(out_path)

# number of steps in gif
num_frames = 20

#### build gif ####

data = np.array([
    [-3.5, 5],
    [-2.5, 4],
    [-1, 2.5],
    [1, 4.5],
    [3, 3],
    [6, 6],
    [8, 10],
    [10, 1]
])
# shift over to start at x=0
data[:, 0] += 3.5

# save plot of signal
ts = Signal(data[:, 1], times=data[:, 0])
fig, ax = plt.subplots()
ts.plot(ax)
plt.savefig(os.path.join(out_path, 'signal.png'), bbox_inches='tight', dpi=300)
plt.close()

# run signal persitence
ts.make_pers(cutoff=-1)
pers = ts.pers.diagram


# hardcoded pause at the first persistence value
middle_pause = pers.death.values.min()

# the first part of the sweep
for i, val in enumerate(np.linspace(data[:, 1].min() - .1,
                                    middle_pause,
                                    num_frames//2)):

    # get components at the cutoff we're running for colors
    if show_components:
        ts = Signal(data[:, 1], times=data[:, 0])
        ts.make_pers(cutoff=val + 1e-6)
        roots = ts.components.values.copy()
        all_roots(roots)
        colors = [plt.cm.Set1(j) for j in roots]
    else:
        colors = single_color

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    viz.plot_signal_threshold(data, tau=val,
                              fig=fig, ax=ax1,
                              title='Signal\nThreshold ($\\tau$): {:.2f}'.format(val),
                              below_tau_color=colors,
                              save=False, show=False)
    # plot persistence values at or below val
    viz.plot_persistence(pers.loc[pers.death <= val, :],
                         bounds=(-0.1, pers.loc[:, 'death'].values.max() + 1),
                         fig=fig, ax=ax2, figsize=(4, 4),
                         save=False, show=False)
    plt.savefig(os.path.join(out_path, 'signal_sweep_pre_{:03}.png'.format(i)), bbox_inches='tight')

# the rest of the sweep
for i, val in enumerate(np.linspace(middle_pause,
                                    data[:, 1].max(),
                                    num_frames//2)):
    # get components at the cutoff we're running for colors
    if show_components:
        ts = Signal(data[:, 1], times=data[:, 0])
        ts.make_pers(cutoff=val + 1e-6)
        roots = ts.components.values.copy()
        all_roots(roots)
        # colors = [plt.cm.Set1(j) for j in roots]
        colors = roots
    else:
        colors = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    viz.plot_signal_threshold(data, tau=val,
                              fig=fig, ax=ax1,
                              title='Signal\nThreshold ($\\tau$): {:.2f}'.format(val),
                              below_tau_color=colors,
                              save=False, show=False)
    # plot persistence values at or below val
    viz.plot_persistence(pers.loc[pers.death <= val, :],
                         bounds=(-0.1, pers.loc[:, 'death'].values.max() + 1),
                         fig=fig, ax=ax2, figsize=(4, 4),
                         save=False, show=False)
    plt.savefig(os.path.join(out_path, 'signal_sweep_post_{:03}.png'.format(i)), bbox_inches='tight')

# make the gif
gif_name = 'signal_sweep'
if show_components:
    gif_name += '_components'
os.system('convert -delay 30> -loop 0 {} -delay 300 {} -delay 30 {} -delay 300 {} {}.gif'.format(
    os.path.join(out_path, 'signal_sweep_pre_*'),
    sorted(glob(os.path.join(out_path, 'signal_sweep_pre_*')))[-1],
    os.path.join(out_path, 'signal_sweep_post_*'),
    sorted(glob(os.path.join(out_path, 'signal_sweep_post_*')))[-1],
    os.path.join(out_path, gif_name)
  )
)