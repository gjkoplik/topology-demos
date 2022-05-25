"""
Gary Koplik
gary<dot>koplik<at>geomdata<dot>com
October, 2019
holoviews_viz.py

Replicating matplotlib visualizations in holoviews
(holoviews gives us the ability to stitch widgets together without a python backend)
"""

from bokeh.resources import INLINE, CDN
import holoviews as hv
hv.extension('matplotlib')
import matplotlib.pyplot as plt
import numpy as np
import os
import panel as pn

def viz_disks_hv(data, radius, padding=1, disk_colors=0,
                 cmap=None, vmin=None, vmax=None):
    """
    visualize disks around points of a certain radius with holoviews (matplotlib backend)

    :param data (np.array shape (n, 2)): array of n points
    :param radius (float > 0): radius of disks around each point
    :param padding (float > 0, default 1): how much to pad boundary around points
    :param disk_colors (float or list-like, default 0): float values for color for disk
        if single float input for color, will use same color for all disks
        if list input for color, will expect color for every data point
    :param cmap (matplotlib.cm colormap, default `None`): cmap to use for disks
        default `None` uses `matplotlib.pyplot.cm.tab10`
    :param vmin (float, default `None`): min value for `cmap`
        default `None` uses `min(disk_colors)`
    :param vmax (float, default `None`): max value for `cmap`
        default `None` uses `max(disk_colors) + 1e-3` (add epsilon in case user puts single float for `disk_colors`
    :return: holoviews.core.overlay.Overlay instance of points and disks
    """

    # coax list input to 1d array
    if type(disk_colors) == list:
        disk_colors = np.array(disk_colors)
    # color for every point
    if type(disk_colors) == np.ndarray:
        assert disk_colors.shape[0] == data.shape[0]
    # coax float to array of colors for each point
    if type(disk_colors) != list and type(disk_colors) != np.ndarray:
        disk_colors = np.array([disk_colors] * data.shape[0])

    if cmap is None:
        cmap = plt.cm.tab10

    points = \
        hv.Points(data).opts(aspect='equal',
                             padding=padding,
                             color='darkorange',
                             edgecolor='black',
                             s=50)

    disks = \
        hv.Polygons([{('x', 'y'): hv.Ellipse(data[i, 0], data[i, 1],
                                             (radius*2, radius*2)).array(),
                      'z': disk_colors[i]} for i in range(data.shape[0])], vdims='z').\
        opts(cmap=cmap,
             edgecolor=None,
             alpha=0.9).\
        redim(z=dict(range=(vmin, vmax)))

    return (disks * points).\
        opts(fig_inches=6,
             xlabel='',
             ylabel='',
             title='Growing Disks Around Each Point'
         )


def viz_persistence_hv(pers_data, threshold=None, padding=1, alpha=0.7, label=''):
    """
    visualize persistence diagram with holoviews (matplotlib backend)

    :param pers_data (pd.DataFrame): 0d persistence info as called by
        `pc = multidim.PointCloud(data, max_length=-1).make_pers0()` then `pc.pers0.diagram`
    :param threshold (float, default `None`): cutoff to run up to (and plot red horizontal line at)
        default `None` does not plot any cutoff line and includes all values in `pers_data`
    :param padding (float > 0, default 1): padding around edges (padding above top pers value will be double this)
    :param alpha (float in [0, 1], default 0.7): alpha value of points
    :param label (str, default ''): label for points if legend is created
        default '' will not create a legend, but any other str will trigger legend creation
    :return: holoviews.core.overlay.Overlay instance of persistence values, a 45 degree line, and a threshold line
    """
    # persistence values
    if threshold is not None:
        points = \
            hv.Points(pers_data.loc[pers_data.death < threshold, ['birth', 'death']], label=label). \
                opts(aspect='equal',
                     alpha=alpha,
                     color='royalblue',
                     s=50
                 )
    else:
        points = \
            hv.Points(pers_data.loc[:, ['birth', 'death']], label=label). \
                opts(aspect='equal',
                     alpha=alpha,
                     color='royalblue',
                     s=50
                     )

    # 45 degree line
    diagonal = \
        hv.Path([(min(0, pers_data.birth.min()) - padding,
                  min(0, pers_data.birth.min()) - padding),
                 (max(pers_data.birth.max(),pers_data.death.max()) + 2 * padding,
                  max(pers_data.birth.max(),pers_data.death.max()) + 2 * padding)]). \
            opts(color='black',
                 title='Persistence Diagram',
                 xlabel='Birth',
                 ylabel='Death'
             )

    # threshold line
    if threshold is not None:
        horizontal_line = \
            hv.Path([(min(0, pers_data.birth.min()) - padding, threshold),
                     (max(pers_data.birth.max(), pers_data.death.max()) + 2 * padding, threshold)]). \
                opts(color='red')

        return (points * diagonal * horizontal_line). \
            opts(fig_inches=6)

    return (points * diagonal). \
        opts(fig_inches=6)

def viz_signal_hv(x, y, title='Signal', padding=1/40, aspect=2.5, **kwargs):
    """
    visualize signal in holoviews (matplotlib backend)

    :param x (list-like): x values for signal
    :param y (list-like): y values for signal
    :param title (str, default 'Signal'): title for plot
    :param padding (float > 0, default 1/40): padding around signal to keep away from axes
    :param aspect (float > 0, default 2.5): width to height ratio for plot
    :param kwargs (dict): other options to modify figure
    :return: holoviews.element.chart.Curve instance of the signal
    """

    # keep some defaults when excluded from kwargs
    options = {'fig_inches': 6,
               "color": 'royalblue',
               'xlabel': ' ',
               'ylabel': ' '}
    # update anything changed by kwargs
    options.update(kwargs)

    curve = \
        hv.Curve(zip(x, y)). \
            opts(aspect=aspect,
                 title=title,
                 padding=padding,
                 **options)

    return curve



def make_hv_widget_one_variable(hv_list, values, name,
                                save=False, path='./figures/',
                                file_name='hv_widget.html', dpi=150):
    """
    turn a list of matplotlib-style holoviews figures varied along a single variable
        into a holoviews.core.spaces.HoloMap instance
        that can be saved as an html widget (without a Python backend)

    :param hv_list (list): holoviews figures to string together in widget
    :param values (list-like): corresponding values for each hv figure
    :param name: name of widget for which we are changing `values`
    :param save (bool, default False): whether to save widget
    :param path (str, default './figures'): where to save the html widget
    :param file_name (str, default 'hv_widet.html'): name of saved html widget
    :param dpi (int > 0, default 150): dpi of each image embedded in html widget
    :return: holoviews.core.spaces.HoloMap instance
    """

    # how the figures render in output widget
    renderer = hv.plotting.mpl.MPLRenderer.instance(dpi=dpi)

    # build holomap
    holomap = hv.HoloMap(dict(zip(values,
                                  hv_list)),
                         kdims=name)
    if save:
        hv.save(holomap, os.path.join(path, file_name))
    return holomap

def make_panel_widget_one_variable(viz_function, values, name, start_value=None,
                                   title='Panel Widget', inline=False,
                                   save=False, path='./figures/',
                                   file_name='panel_widget.html'):
    """
    run a one-parameter `viz_function` for all `values`, string together into a Panel app,
        and save as an html widget (without a Python backend)

    :param viz_function (function): one-parameter function that generates some visualization.
        Pre-computes figures for all `values`
    :param values (list-like): values to run `viz_function` on
    :param name: name of widget for which we are changing `values`
    :param start_value (value contained in `values`, default `None`): value on slider used upon initialization of html widget
        default `None` uses the first term in `values`
    :param title (str, default 'Panel Widget'): name of tab of html widget when opened in browser
    :param inline (bool, default False): whether to download all needed bokeh js and css information into the widget
        or pull information from the internet (at least in theory, given
            https://bokeh.pydata.org/en/latest/docs/reference/resources.html)
        True should in theory make the resulting widget work independent of an internet connection
            (at the cost of a larger html file)
        However, in practice, the widget seems to be working regardless,
            so the default is for now False to keep the resulting widget smaller
    :param save (bool, default False): whether to save widget
    :param path (str, default './figures'): where to save the html widget
    :param file_name (str, default 'panel_widet.html'): name of saved html widget
    :return: panel.layout.Column instance
        (which can be viewed in a jupyter notebook as is or in a new tab with `<instance_name>.show()`
    """

    if start_value is None:
        starting_widget_val = values[0]
    else:
        starting_widget_val = start_value

    slider = pn.widgets.DiscreteSlider(name=name, options=list(values), value=starting_widget_val)

    @pn.depends(slider.param.value)
    def callback(value):
        """
        run the callback according to the specified `viz_function`

        :param value (a value in `values`):
        :return: some visualization
        """
        return viz_function(value)

    app = pn.Column(callback, pn.Spacer(), slider)

    if inline:
        resources = INLINE
    else:
        resources = CDN

    if save:
        app.save(os.path.join(path, file_name),
                 resources=resources, embed=True, title=title)

    return app