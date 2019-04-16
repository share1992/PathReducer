import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter


def _plot_1d(x):
    plt.figure(figsize=(8,4))
    ax = plt.axes[0]
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_xlabel('Frame', fontsize=16)
    ax.set_ylabel('PC', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.plot(range(len(x)), x)
    return plt

def _plot_2d(x, y, smooth=False, fig=None):
    """
    Plots a 2D plot of x vs y. Colors the datapoints from yellow to purple 
    to show time progression. Can be called multiple times by passing the return
    figure as input.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2)
        ax0 = fig.add_subplot(gs[0])
        ax0.grid(True, alpha=0.4)
        first_plot = True
    else:
        ax0 = fig.axes[0]
        first_plot = False

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

    # Update limits
    xrange_ = x_i.max() - x_i.min()
    yrange_ = y_i.max() - y_i.min()
    xmin = x_i.min()-0.1*xrange_
    xmax = x_i.max()+0.1*xrange_
    ymin = y_i.min()-0.1*yrange_
    ymax = y_i.max()+0.1*yrange_
    if not first_plot:
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        xmin = min(xmin, xlim[0])
        xmax = max(xmax, xlim[1])
        ymin = min(ymin, ylim[0])
        ymax = max(ymax, ylim[1])
    ax0.set_xlim([xmin, xmax])
    ax0.set_ylim([ymin, ymax])
    return fig

    #if lengths is None:
    #    # ax0
    #    # fit spline
    #    tck, u = interpolate.splprep([x, y], k=1, s=0.0)
    #    x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    #    # Gradient color change magic
    #    z = np.linspace(0.0, 1.0, x_i.shape[0])
    #    points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
    #    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #    lc = LineCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)
    #    ax0.add_collection(lc)

    #    if x2 is not None and y2 is not None:
    #        tck2, u2 = interpolate.splprep([x2, y2], k=1, s=0.0)
    #        x_i2, y_i2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

    #        # Gradient color change magic
    #        z = np.linspace(0.0, 1.0, x_i2.shape[0])
    #        points2 = np.array([x_i2, y_i2]).T.reshape(-1, 1, 2)
    #        segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    #        lc2 = LineCollection(segments2, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8,
    #                            linewidth=2)
    #        ax0.add_collection(lc2)

    #        x_i = np.concatenate((x_i, x_i2))
    #        y_i = np.concatenate((y_i, y_i2))

    #    # plotting
    #    if same_axis:
    #        max_ = max(x_i.max(), y_i.max())
    #        min_ = min(x_i.min(), y_i.min())
    #        range_ = max_ - min_
    #        ax0.set_xlim([min_-0.1*range_, max_+0.1*range_])
    #        ax0.set_ylim([min_-0.1*range_, max_+0.1*range_])
    #    else:
    #        xrange_ = x_i.max() - x_i.min()
    #        yrange_ = y_i.max() - y_i.min()
    #        ax0.set_xlim([x_i.min()-0.1*xrange_, x_i.max()+0.1*xrange_])
    #        ax0.set_ylim([y_i.min()-0.1*yrange_, y_i.max()+0.1*yrange_])

    #    # time plots
    #    # ax1.plot(range(len(x)), x)
    #    # ax1.plot(range(len(y)), y)

    #elif lengths is not None:
    #    for i in range(len(lengths)):
    #        if i == 0:
    #            xseg = x[0:lengths[i]]
    #            yseg = y[0:lengths[i]]

    #            # ax0
    #            # fit spline
    #            tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
    #            x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    #            # Gradient color change magic
    #            z = np.linspace(0.0, 1.0, x_i.shape[0])
    #            points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
    #            segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #            lc = LineCollection(segments, array=z, cmap='viridis',
    #                                norm=plt.Normalize(0.0, 1.0), alpha=0.8)
    #            ax0.add_collection(lc)

    #        else:
    #            start_index = sum(lengths[:i])
    #            end_index = sum(lengths[:(i+1)])
    #            # print("start_index = %s" % start_index)
    #            # print("end_index = %s" % end_index)
    #            xseg = x[start_index:end_index]
    #            yseg = y[start_index:end_index]

    #            # ax0
    #            # fit spline
    #            tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
    #            x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

    #            # Gradient color change magic
    #            z = np.linspace(0.0, 1.0, x_i.shape[0])
    #            points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
    #            segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #            lc = LineCollection(segments, array=z, cmap='viridis',
    #                                norm=plt.Normalize(0.0, 1.0), alpha=0.8)
    #            ax0.add_collection(lc)

    #    # Setting axis limits
    #    if same_axis:
    #        max_ = max(x.max(), y.max())
    #        min_ = min(x.min(), y.min())
    #        range_ = max_ - min_
    #        ax0.set_xlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
    #        ax0.set_ylim([min_ - 0.1 * range_, max_ + 0.1 * range_])
    #    else:
    #        xrange_ = x.max() - x.min()
    #        yrange_ = y.max() - y.min()
    #        ax0.set_xlim([x.min() - 0.1 * xrange_, x.max() + 0.1 * xrange_])
    #        ax0.set_ylim([y.min() - 0.1 * yrange_, y.max() + 0.1 * yrange_])

    ## Adding new_data, if it exists
    #if new_data is not None:
    #    tck_new, u_new = interpolate.splprep([new_data[0], new_data[1]], k=1, s=0.0)
    #    x_i_new, y_i_new = interpolate.splev(np.linspace(0, 1, 1000), tck_new)

    #    # Gradient color change magic
    #    z_new = np.linspace(0.0, 1.0, x_i_new.shape[0])
    #    points_new = np.array([x_i_new, y_i_new]).T.reshape(-1, 1, 2)
    #    segments_new = np.concatenate([points_new[:-1], points_new[1:]], axis=1)
    #    lc_new = LineCollection(segments_new, array=z_new, cmap='viridis', norm=plt.Normalize(0.0, 1.0),
    #                        linestyle="solid", linewidth=2.5)
    #    lca_new = LineCollection(segments_new, color='black', linestyle="solid", linewidth=5.0)
    #    ax0.add_collection(lca_new)
    #    ax0.add_collection(lc_new)

    #    x_i_new = np.concatenate((x_i, x_i_new))
    #    y_i_new = np.concatenate((y_i, y_i_new))

    #    xrange_ = x_i_new.max() - x_i_new.min()
    #    yrange_ = y_i_new.max() - y_i_new.min()
    #    ax0.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
    #    ax0.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])

def _plot_3d():
    # Do 3D plot

    ax1 = fig.add_subplot(gs[1], projection='3d')

    ax1.set_xlabel('P.C. 1', fontsize=16, labelpad=10)
    ax1.set_ylabel('P.C. 2', fontsize=16, labelpad=10)
    ax1.set_zlabel('P.C. 3', fontsize=16, labelpad=10)
    ax1.tick_params(axis='both', labelsize=12)#, pad=5)
    ax1.ticklabel_format(style='sci', scilimits=(-3,3))
    # ax1.set_title('Top Three Principal Components', fontsize=18, fontstyle='italic')
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.dist = 9
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Malonaldehyde
    # ax1.scatter(x[0], y[0], y1[0], edgecolors='k', facecolors='none', s=50, zorder=10)
    # ax1.scatter(x[12], y[12], y1[12], edgecolors='k', facecolors='none', s=50, zorder=10)
    # ax1.scatter(x[24], y[24], y1[24], edgecolors='k', facecolors='none', s=50, zorder=10)
    # SN2
    # ax1.scatter(x[0],y[0],y1[0], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
    # ax1.scatter(x[15],y[15],y1[15], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
    # ax1.scatter(x[24],y[24],y1[24], edgecolors= 'k', facecolors = 'none', s=50, zorder=10)
    # ax1.scatter(x[103],y[103],y1[103], edgecolors= 'k', facecolors = 'none', s=50, zorder=10)

    if lengths is None or sum(lengths) == 0:

        # ax1
        # fit spline
        tck, u = interpolate.splprep([x, y, y1], k=1, s=0.0)
        x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # Gradient color change magic
        z = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8,
                              linewidth=2)
        ax1.add_collection(lc)

        if x2 is not None and y2 is not None and y12 is not None:
            tck, u = interpolate.splprep([x2, y2, y12], k=1, s=0.0)
            x_i2, y_i2, z_i2 = interpolate.splev(np.linspace(0, 1, 1000), tck)

            # Gradient color change magic
            z2 = np.linspace(0.0, 1.0, x_i2.shape[0])
            points = np.array([x_i2, y_i2, z_i2]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc2 = Line3DCollection(segments, array=z2, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8,
                                  linewidth=2)
            ax1.add_collection(lc2)

            x_i = np.concatenate((x_i, x_i2))
            y_i = np.concatenate((y_i, y_i2))
            z_i = np.concatenate((z_i, z_i2))

        # Setting axis limits
        if same_axis:
            max_ = max(x_i.max(), y_i.max(), z_i.max())
            min_ = min(x_i.min(), y_i.min(), z_i.min())
            range_ = max_ - min_
            ax1.set_xlim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_ylim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_zlim([min_-0.1*range_, max_+0.1*range_])
        else:
            xrange_ = x_i.max() - x_i.min()
            yrange_ = y_i.max() - y_i.min()
            zrange_ = z_i.max() - z_i.min()
            ax1.set_xlim([x_i.min()-0.1*xrange_, x_i.max()+0.1*xrange_])
            ax1.set_ylim([y_i.min()-0.1*yrange_, y_i.max()+0.1*yrange_])
            ax1.set_zlim([z_i.min()-0.1*zrange_, z_i.max()+0.1*zrange_])

    elif lengths is not None:
        for i in range(len(lengths)):
            if i == 0:
                xseg = x[0:lengths[i]]
                yseg = y[0:lengths[i]]
                zseg = y1[0:lengths[i]]

                # ax1
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg, zseg], k=1, s=0.0)
                x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                z = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax1.add_collection(lc)

            else:
                start_index = sum(lengths[:i])
                end_index = sum(lengths[:(i+1)])
                # print("start_index = %s" % start_index)
                # print("end_index = %s" % end_index)
                xseg = x[start_index:end_index]
                yseg = y[start_index:end_index]
                zseg = y1[start_index:end_index]

                # ax1
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg, zseg], k=1, s=0.0)
                x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                z = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax1.add_collection(lc)

        # Setting axis limits
        if same_axis:
            max_ = max(x.max(), y.max(), y1.max())
            min_ = min(x.min(), y.min(), y1.min())
            range_ = max_ - min_
            ax1.set_xlim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_ylim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_zlim([min_-0.1*range_, max_+0.1*range_])
        else:
            xrange_ = x.max() - x.min()
            yrange_ = y.max() - y.min()
            zrange_ = y1.max() - y1.min()
            ax1.set_xlim([x.min()-0.1*xrange_, x.max()+0.1*xrange_])
            ax1.set_ylim([y.min()-0.1*yrange_, y.max()+0.1*yrange_])
            ax1.set_zlim([y1.min()-0.1*zrange_, y1.max()+0.1*zrange_])

    # Adding new_data, if it exists
    if new_data is not None:
        tck_new, u_new = interpolate.splprep([new_data[0], new_data[1], new_data[2]], k=1, s=0.0)
        x_i_new, y_i_new, z_i_new = interpolate.splev(np.linspace(0, 1, 1000), tck_new)

        # Gradient color change magic
        z_new = np.linspace(0.0, 1.0, x_i_new.shape[0])
        points_new = np.array([x_i_new, y_i_new, z_i_new]).T.reshape(-1, 1, 3)
        segments_new = np.concatenate([points_new[:-1], points_new[1:]], axis=1)
        lc_new = Line3DCollection(segments_new, array=z_new, cmap='viridis', norm=plt.Normalize(0.0, 1.0),
                            linestyle="solid", linewidth=2.5)
        lca_new = Line3DCollection(segments_new, color='black', linestyle="solid", linewidth=5.0)
        ax1.add_collection(lca_new)
        ax1.add_collection(lc_new)

        x_i_new = np.concatenate((x_i, x_i_new))
        y_i_new = np.concatenate((y_i, y_i_new))
        z_i_new = np.concatenate((z_i, z_i_new))

        xrange_ = x_i_new.max() - x_i_new.min()
        yrange_ = y_i_new.max() - y_i_new.min()
        zrange_ = z_i_new.max() - z_i_new.min()
        ax1.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
        ax1.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])
        ax1.set_zlim([z_i_new.min() - 0.1 * zrange_, z_i_new.max() + 0.1 * zrange_])

# TODO support for multiple plots on top of each other
def colorplot(x, smooth=False, image_name=None):
    """
    Create a 1-3D plot.

    :param x: Data to plot. Should be of shape (n,m), with m being between 1 and 3.
    :type x: array
    :param smooth: Whether or not to smooth the line between data points with a spline.
    :type smooth: bool
    :param image_name: Full location to store the image. If `None`, the image will be displayed
                       instead.
    :type image_name: string
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
                fig = _plot_1d(array.ravel(), smooth=smooth, fig=fig)
        else:
            fig = _plot_1d(x.ravel(), smooth=smooth)
    elif m == 2:
        if multi_array:
            for array in x:
                fig = _plot_2d(array[:,0] + 1, array[:,1], smooth=smooth, fig=fig)
        else:
            fig = _plot_2d(x[:,0], x[:,1], smooth=smooth)
    elif m == 3:
        if multi_array:
            for array in x:
                fig = _plot_3d(array[:,0], array[:,1], array[:,2], fig=fig)
        else:
            fig = _plot_3d(x[:,0], x[:,1], x[:,2])
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
