"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
July, 2019
brain_example.py

Going through brain examples hosted at:
    https://gitlab.com/alexpieloch/PersistentHomologyAnalysisofBrainArteryTrees
in the context of the paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5026243/pdf/nihms777844.pdf
"""

import datetime
import dateutil.tz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from viz import viz

#### hardcoded variables ####

# number of brain cases in dataset
num_cases = 109

# age cutoff of old vs young
age_cutoff = 50

# path to 1d persistence information
pers_1_path = './sandbox/data_PersistenceDiagrams_H1PersistenceDiagrams.mat'

# color for below age cutoff in aggregate 1d persistence diagram figures
below_color = 'C1'

# color for above age cutoff in aggregate 1d persistence diagram figures
above_color = 'C0'

# alpha values for aggregate 1d persistence diagram figures
alpha = 0.02

# case number of brains files to use (ints in {1, 2, ..., 109})
brain_numbers = [3, 29]

# path to brain files
brain_paths = [f'./sandbox/res/decorated_tree.aca.Case{i}.mat' for i in brain_numbers]

# path to metadata files (e.g. Age, sex)
metadata_path = './sandbox/CaseInformation.mat'

# angle above XY plane when we rotate around
angle_above_XY = 45

# size of sample number of points to draw from each brain file for plotting purposes
num_pts = 50000

# how many degrees to skip between frames
#  NOTE: works well for continuous gif only if `skip` divides 360 with no remainder
skip = 10

# color of brains in output
color = 'C0'

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

out_path = os.path.join('./figures', "brain_example_{}".format(datetime_stamp))
os.mkdir(out_path)

# Ages corresponding to each chosen brain
metadata = loadmat(metadata_path)

# all ages
age_values = metadata['Ages'].flatten()

# cases start on index 1, python is index 0
brain_ages = metadata['Ages'].flatten()[np.array(brain_numbers)-1]

brain_out_paths = [os.path.join(out_path, f'brain_age_{age}_case_{case}')
                   for age, case in zip(brain_ages, brain_numbers)]

for brain_out_path in brain_out_paths:
    os.mkdir(brain_out_path)

#### compare brain persistence diagrams ####

# read in 1d pers information
pers1_values = loadmat(pers_1_path)['H1Array']

# case specific 1d persistence diagrams
for index, age, case in zip(np.array(brain_numbers)-1, brain_ages, brain_numbers):
    diagram = pers1_values[index][0]
    # apparently at least one brain with no 1d pers values
    if diagram.size > 0:
        # infinite pers values are -1 (remove for now)
        rm_negative_one = np.delete(diagram, np.where(diagram[:, 1] == -1)[0], 0)
        viz.plot_persistence(pd.DataFrame(rm_negative_one, columns=['birth', 'death']),
                             alpha=0.5, show=False, color=above_color,
                             title=f'1d Persistence Values Age {age}',
                             save=True, path=out_path, name=f'pers1_age_{age}_case_{case}')
    else:
        print(f"no 1d persistence values for case {case} age {age}")


# break cases into above and below age cutoff
age_bool = age_values < age_cutoff

below_cutoff_idxs = np.arange(num_cases)[age_bool]

above_cutoff_idxs = np.arange(num_cases)[np.logical_not(age_bool)]

df_below = pd.DataFrame(columns=['birth', 'death'])
for index in below_cutoff_idxs:
    diagram = pers1_values[index][0]
    # apparently at least one brain with no 1d pers values
    if diagram.size > 0:
        # infinite pers values are -1 (remove for now)
        rm_negative_one = np.delete(diagram, np.where(diagram[:, 1] == -1)[0], 0)
        df_below = pd.concat([df_below, pd.DataFrame(rm_negative_one, columns=['birth', 'death'])])
# below age cutoff part of above and below figure
fig, ax = viz.plot_persistence(df_below, alpha=alpha, show=False, color=below_color)

# below age cutoff figure
viz.plot_persistence(df_below, alpha=alpha, show=False,
                     title='1d Persistence Below Age Cutoff',
                     color=below_color,
                     save=True, path=out_path,
                     name='pers_1d_below_age_cutoff.png',
                     dpi=300)

df_above = pd.DataFrame(columns=['birth', 'death'])
for index in above_cutoff_idxs:
    diagram = pers1_values[index][0]
    # apparently at least one brain with no 1d pers values
    if diagram.size > 0:
        # infinite pers values are -1 (remove for now)
        rm_negative_one = np.delete(diagram, np.where(diagram[:, 1] == -1)[0], 0)
        df_above = pd.concat([df_above, pd.DataFrame(rm_negative_one, columns=['birth', 'death'])])

# above age cutoff figure
viz.plot_persistence(df_above, alpha=alpha, show=False,
                     title='1d Persistence Above Age Cutoff',
                     color=above_color,
                     save=True, path=out_path,
                     name='pers_1d_above_age_cutoff.png',
                     dpi=300)

# above and below age cutoff one figure
viz.plot_persistence(df_above, alpha=alpha, fig=fig, ax=ax,
                     title='1d Persistence Both Age Cutoffs',
                     color=above_color,
                     save=True, path=out_path,
                     name='pers_1d_both_age_cutoffs.png',
                     dpi=300)

# plotting number of 1d pers values vs age
point_clouds = np.array([i[0] for i in pers1_values])
sizes = [i.size for i in point_clouds]
# pull out empty 1d pers diagrams
empty_diagram_idxs = np.array(sizes) == 0
point_clouds = point_clouds[np.logical_not(empty_diagram_idxs)]
# number of 1d pers values in each diagram
num_points = np.array([i.shape[0] for i in point_clouds])

# plot with outliers
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(num_points, age_values[np.logical_not(empty_diagram_idxs)])
ax.set_ylabel('Age')
ax.set_xlabel('Number of 1d Persistence Values')
ax.set_title('A Small But Statistically Significant Relationship')
plt.savefig(os.path.join(out_path, 'num_pers_vals_vs_age_with_outliers.png'),
            dpi=200, bbox_inches='tight')

# plot without outliers
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(num_points, age_values[np.logical_not(empty_diagram_idxs)])
ax.set_ylabel('Age')
ax.set_xlabel('Number of 1d Persistence Values')
ax.set_xlim(0, 1000)
ax.set_title('A Small But Statistically Significant Relationship\n(Without Outliers)')
plt.savefig(os.path.join(out_path, 'num_pers_vals_vs_age_without_outliers.png'),
            dpi=200, bbox_inches='tight')


#### build brain gifs ####

# loop over brains
for brain_path, brain_number, brain_age, brain_out_path in \
        zip(brain_paths, brain_numbers, brain_ages, brain_out_paths):

    # read in data
    mat = loadmat(brain_path)
    points = np.array([mat['T'][0, 0][1][:, k][0][0][0] for k in range(mat['T'][0, 0][1].shape[1])])
    # take subset of brain indices (don't need ~100k points for good pic)
    np.random.seed(27705)
    sample_idxs = np.random.choice(points.shape[0], num_pts, replace=False)
    sample = points[sample_idxs, :]
    X = sample[:, 0]
    Y = sample[:, 1]
    Z = sample[:, 2].reshape(-1, 1)
    # build images for each brain
    for i, rotation in enumerate(np.arange(0, 360, skip)):
        rotation_XY = rotation

        include_colorbar = False

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(angle_above_XY, rotation_XY)
        #     for color in np.unique(colors):
        #         sub_sample = sample[np.where(colors==color)[0], :]
        #         ax.plot(sub_sample[:, 0],
        #                 sub_sample[:, 1],
        #                 sub_sample[:, 2])
        surface = ax.scatter(X, Y, Z,
                             c=color,
                             alpha=0.1,
                             s=1,
                             antialiased=False)
        ax.set_title(f'Brain Age {brain_age}')
        if include_colorbar:
            cb = fig.colorbar(surface)
            cb.ax.set_title('Height Color')
        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.savefig(os.path.join(brain_out_path, 'brain_{:03}'.format(i)), dpi=100, bbox_inches='tight')
        plt.close()

    # make gif
    os.system('convert -loop 0 {} {}.gif'.format(
        os.path.join(brain_out_path, 'brain_*'),
        os.path.join(out_path, f'brain_age_{brain_age}_case_{brain_number}')))


