import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter

def _update_limits(ax, x, y, z=None, add_xrange=True, first_plot=True, same_axis=False):
    # Update limits
    yrange_ = y.max() - y.min()
    xmin = x.min()
    xmax = x.max()
    if add_xrange:
        xrange_ = x.max() - x.min()
        xmin -= 0.1*xrange_
        xmax += 0.1*xrange_
    ymin = y.min()-0.1*yrange_
    ymax = y.max()+0.1*yrange_
    if z is not None:
        zrange_ = z.max() - z.min()
        zmin = x.min() - 0.1*zrange_
        zmax = x.max() + 0.1*zrange_
    if not first_plot:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmin = min(xmin, xlim[0])
        xmax = max(xmax, xlim[1])
        ymin = min(ymin, ylim[0])
        ymax = max(ymax, ylim[1])
        if z is not None:
            zlim = ax.get_zlim()
            zmin = min(zmin, zlim[0])
            zmax = max(zmax, zlim[1])
    if same_axis:
        if z is None:
            xmin = min(xmin, ymin)
            ymin = xmin
            xmax = max(xmax, ymax)
            ymax = xmax
        else:
            xmin = min(xmin, ymin, zmin)
            ymin = xmin
            zmin = xmin
            xmax = max(xmax, ymax, zmax)
            ymax = xmax
            zmax = xmax

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if z is not None:
        ax.set_zlim([zmin, zmax])
    return ax


def _plot_1d(x, smooth=False, fig=None, same_axis=False):
    """
    Plots a 1D plot of x vs the frame number. Colors the datapoints from yellow to purple 
    to show time progression. Can be called multiple times by passing the return
    figure as input.
    """
    if fig is None:
        fig = plt.figure(figsize=(8,4))
        first_plot = True
    else:
        first_plot = False
    ax = fig.gca()
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_xlabel('Frame', fontsize=16)
    ax.set_ylabel('PC', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    if smooth:
        k = 2
    else:
        k = 1

    tck, u = interpolate.splprep([np.arange(x.shape[0]), x], k=k, s=0.0)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    # Gradient color change magic
    z = np.linspace(0.0, 1.0, x_i.shape[0])
    points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)
    ax.add_collection(lc)

    ax = _update_limits(ax, x_i, y_i, add_xrange=False, first_plot=first_plot, same_axis=same_axis)

    return fig

def _plot_2d(x, y, smooth=False, fig=None, same_axis=False):
    """
    Plots a 2D plot of x vs y. Colors the datapoints from yellow to purple 
    to show time progression. Can be called multiple times by passing the return
    figure as input.
    """
    if fig is None:
        fig = plt.figure(figsize=(5, 5))
        #gs = GridSpec(1, 2)
        #ax0 = fig.add_subplot(gs[0])
        first_plot = True
    else:
        first_plot = False
    ax0 = fig.gca()
    ax0.grid(True, alpha=0.4)

    ax0.set_xlabel('PC1', fontsize=16)
    ax0.set_ylabel('PC2', fontsize=16)
    ax0.tick_params(axis='both', labelsize=12)

    if smooth:
        k = 2
    else:
        k = 1

    tck, u = interpolate.splprep([x, y], k=k, s=0.0)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    # Gradient color change magic
    z = np.linspace(0.0, 1.0, x_i.shape[0])
    points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)
    ax0.add_collection(lc)

    ax0 = _update_limits(ax0, x_i, y_i, first_plot=first_plot, same_axis=same_axis)

    return fig

def _plot_3d(x, y, z, smooth=False, fig=None, same_axis=False):
    """
    Plots a 3D plot of x, y, z. Colors the datapoints from yellow to purple 
    to show time progression. Can be called multiple times by passing the return
    figure as input.
    """
    if fig is None:
        fig = plt.figure(figsize=(5, 5))
        #gs = GridSpec(1, 2)
        #ax0 = fig.add_subplot(gs[0])
        ax0 = fig.add_subplot(111, projection='3d')
        first_plot = True
    else:
        ax0 = fig.axes[0]
        first_plot = False
    ax0.grid(True, alpha=0.4)

    ax0.set_xlabel('PC1', fontsize=16, labelpad=10)
    ax0.set_ylabel('PC2', fontsize=16, labelpad=10)
    ax0.set_zlabel('PC3', fontsize=16, labelpad=10)
    ax0.tick_params(axis='both', labelsize=12)
    #ax0.ticklabel_format(style='sci', scilimits=(-3,3))

    if smooth:
        k = 2
    else:
        k = 1

    tck, u = interpolate.splprep([x, y, z], k=k, s=0.0)
    x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    # Gradient color change magic
    t = np.linspace(0.0, 1.0, x_i.shape[0])
    points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=t, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)
    ax0.add_collection(lc)

    ax0 = _update_limits(ax0, x_i, y_i, z_i, first_plot=first_plot, same_axis=same_axis)

    return fig

# TODO support for multiple subplots
def colorplot(x, smooth=False, image_name=None, same_axis=False):
    """
    Create a 1-3D plot.

    :param x: Data to plot. Should be of shape (n,m), with m being between 1 and 3.
    :type x: array
    :param smooth: Whether or not to smooth the line between data points with a spline.
    :type smooth: bool
    :param image_name: Full location to store the image. If `None`, the image will be displayed
                       instead.
    :type image_name: string
    :param same_axis: Force the same limits to be used on all axes in the plot
    :type same_axis: bool
    """
    def get_dimensionality(x):
        if isinstance(x, list) or (isinstance(x, np.ndarray) and (x.dtype == object or x.ndim == 3)):
            if len(x) > 0 and isinstance(x[0], np.ndarray):
                array_sizes = [array.ndim for array in x]
                if len(set(array_sizes)) > 1:
                    raise SystemExit("Error. Expected and 1D or 2D array or list of arrays of similar shape.")
                if array_sizes[0] > 2 or array_sizes[0] < 1:
                    raise SystemExit("Error. Expected an 1D or 2D array. Got %dD." % x.ndim)
                if array_sizes[0] == 1:
                    m = 1
                else:
                    dimension_sizes = [array.shape[1] for array in x]
                    if len(set(dimension_sizes)) > 1:
                        raise SystemExit("Error. Inconsistent dimensionality of arrays.")

                    m = x[0].shape[1]
                return m, True
            else:
                raise SystemExit("Error. Expected an 1D or 2D array or list of arrays of similar shape.")
        elif x.ndim > 2 or x.ndim == 0:
            raise SystemExit("Error. Expected an 1D or 2D array. Got %dD." % x.ndim)
        elif x.ndim == 1:
            m = 1
        else:
            m = x.shape[1]

        return m, False

    m, multi_array = get_dimensionality(x)

    fig = None
    if m == 1:
        if multi_array:
            for array in x:
                fig = _plot_1d(array.ravel(), smooth=smooth, fig=fig, same_axis=same_axis)
        else:
            fig = _plot_1d(x.ravel(), smooth=smooth, same_axis=same_axis)
    elif m == 2:
        if multi_array:
            for array in x:
                fig = _plot_2d(array[:,0] + 1, array[:,1], smooth=smooth, fig=fig, same_axis=same_axis)
        else:
            fig = _plot_2d(x[:,0], x[:,1], smooth=smooth, same_axis=same_axis)
    elif m == 3:
        if multi_array:
            for array in x:
                fig = _plot_3d(array[:,0], array[:,1], array[:,2], fig=fig, same_axis=same_axis)
        else:
            fig = _plot_3d(x[:,0], x[:,1], x[:,2], same_axis=same_axis)
    else:
        raise SystemExit("Error. Only 1-3D plots are supported, while the given input \
                had %d dimension." % m)

    fig.tight_layout(pad=5)
    fig.subplots_adjust(top=0.88, wspace=0.02)

    if image_name is None:
        plt.show()
    else:
        plt.savefig(image_name, dpi=600)
        plt.clf()
