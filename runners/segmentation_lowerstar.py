"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
July, 2019
segmentation_lowerstar.py

This example is based on an example Chris Tralie covered for a demo in John Harer's topology class (Sep, 2018).
"""

import datetime
import dateutil.tz
import holoviews as hv
from homology.dim0 import all_roots
import matplotlib.pyplot as plt
import multidim
import numpy as np
import numpy.ma as ma
import os
import PIL
from scipy.ndimage import uniform_filter
from viz import holoviews_viz
from viz import viz

#### hardcoded information ####

path_to_wood_cells = "./Cells.jpg"

uniform_filter_size = 10

# threshold for lower star filtration
thresh = 70

# min and max thresholds for building widget
min_thresh, max_thresh = (0, 150)

# number of unique thresholds to build into holomap
num_thresh_points = 50

# make timestamped folder
now = datetime.datetime.now(dateutil.tz.tzlocal())
datetime_stamp = now.strftime('%Y%m%d_%H%M%S_%f')

# out_dir
out_path = os.path.join('./figures', "segmentation_lowerstar_{}".format(datetime_stamp))
os.mkdir(out_path)


#### build out example ####

# initial wood cells in rgb
cells_original = plt.imread(path_to_wood_cells)
# turn from rgb to grayscale
cells_gray = np.asarray(PIL.Image.fromarray(cells_original).convert('L'), float)
# blur the image and invert (want the cell interiors to be low values relative to cell walls)
cells_gray = -uniform_filter(cells_gray, size=uniform_filter_size)
# make min value 0
cells_gray = cells_gray - cells_gray.min()

# save the blurred gray scale image
viz.plot_image(cells_gray, cmap='Blues_r', include_colorbar=False,
               title='Blurred Image With 1 Dimension of Color',
               origin='upper',
               show=False, save=True,
               name='blurred_grayscale_image.png', path=out_path, dpi=300)

# run lowerstar
lowerstar = multidim.lower_star_for_image(cells_gray,
                                          diagonals_and_faces=False)
# reset lowerstar to build up connections to exact threshold of interest
lowerstar.reset()
# build simplicial complex up to appropriate connectedness
lowerstar.make_pers0(cutoff=thresh, until_connected=(1,1), show_diagonal=True)
# pull out information of how indices are connected
roots = lowerstar.stratum[0]['rep'].values.copy()
all_roots(roots)
roots.shape = cells_gray.shape
# mask values not yet swept up to
mask = ma.masked_where(cells_gray > thresh, roots)

# use same random cmap throughout for image
np.random.seed(27701)
# need at most a random color index for every pixel of `density`
image_cmap = viz.random_cmap(np.arange(cells_gray.size))

# plot the connected components
viz.plot_image(mask, cmap=image_cmap, include_colorbar=False,
               origin='upper',
               title='Connected Components',
               save=True, show=False,
               name='connected_components.png',
               path=out_path, dpi=300)

# find the centers of each connected component
x = []
y = []
for idx in np.unique(mask).data:
    cell_idxs = np.where(roots == idx)
    x.append(cell_idxs[0].mean())
    y.append(cell_idxs[1].mean())

fig, ax = viz.plot_image(cells_original,
                         origin='upper',
                         title="Identified Cells",
                         include_colorbar=False, show=False)
ax.scatter(y, x, c='C1', s=10)
plt.savefig(os.path.join(out_path, "identified_cells.png"), dpi=300, bbox_inches='tight')

# plot the segmentation on top of the image
fig, ax = viz.plot_image(cells_original,
                         origin='upper',
                         title="Segmentation Coverage",
                         include_colorbar=False, show=False)
ax.imshow(mask, cmap=image_cmap)
plt.savefig(os.path.join(out_path, "segmentation_coverage.png"), dpi=300, bbox_inches='tight')

#### widget of different values ####

# store holoviews figures
figures = []

# range of threshold values to check
thresh_values = np.linspace(min_thresh, max_thresh, num_thresh_points)

for i, threshold in enumerate(thresh_values):
    if i % 5 == 0:
        print(f"Now on {i} of {num_thresh_points}")

    # reset lowerstar to build up connections to exact threshold of interest
    lowerstar.reset()
    # build simplicial complex up to appropriate connectedness
    lowerstar.make_pers0(cutoff=threshold, until_connected=(1,1), show_diagonal=True)
    # pull out information of how indices are connected
    roots = lowerstar.stratum[0]['rep'].values.copy()
    all_roots(roots)
    roots.shape = cells_gray.shape
    # mask values not yet swept up to
    mask = ma.masked_where(cells_gray > threshold, roots)

    hv_figure = \
        (hv.RGB(cells_original) * hv.Image(mask).opts(cmap=image_cmap)).\
            opts(title="Segmentation Coverage",
                 xaxis=None, yaxis=None,
                 fig_inches=(6, 6))
    figures.append(hv_figure)

holomap = holoviews_viz.make_hv_widget_one_variable(figures, thresh_values, name="Threshold",
                                                    save=True, path=out_path,
                                                    file_name='segmentation_lowerstar_widget.html')