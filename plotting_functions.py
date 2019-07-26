import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter
import os
import glob


def gradient_color_change_magic(x, y, z=None):
    if z is None:

        # fit spline
        tck, u = interpolate.splprep([x, y], k=1, s=0.0)
        x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # Gradient color change magic
        x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)

    elif z is not None:
        # fit spline
        tck, u = interpolate.splprep([x, y, z], k=1, s=0.0)
        x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

        # Gradient color change magic
        x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
        points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)

    return lc


def colored_line_plot(x, y, z=None, x2=None, y2=None, z2=None, imgname=None, same_axis=False, lengths=None,
                      output_directory=None, points_to_circle=None):
    """
    Create a 2D and/or 3D plot
    """
    # Create figure
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2)
    # Do 2D plot
    ax0 = fig.add_subplot(gs[0])
    ax0.grid(True)

    ax0.set_xlabel('PC1', fontsize=16)
    ax0.set_ylabel('PC2', fontsize=16)
    ax0.tick_params(axis='both', labelsize=12)

    if lengths is None:
        # One trajectory
        lc = gradient_color_change_magic(x, y)
        ax0.add_collection(lc)

        if x2 is not None and y2 is not None:
            lc2 = gradient_color_change_magic(x2, y2)
            ax0.add_collection(lc2)

    else:
        # Multiple concatenated trajectories
        for i in range(len(lengths)):
            if i == 0:
                xseg = x[0:lengths[i]]
                yseg = y[0:lengths[i]]

                lc = gradient_color_change_magic(xseg, yseg)
                ax0.add_collection(lc)

            else:
                start_index = sum(lengths[:i])
                end_index = sum(lengths[:(i+1)])
                xseg = x[start_index:end_index]
                yseg = y[start_index:end_index]

                lc = gradient_color_change_magic(xseg, yseg)
                ax0.add_collection(lc)

    if z is not None:
        # Do 3D plot
        ax1 = fig.add_subplot(gs[1], projection='3d')
        ax1.set_xlabel('PC1', fontsize=16, labelpad=9)
        ax1.set_ylabel('PC2', fontsize=16, labelpad=7)
        ax1.set_zlabel('PC3', fontsize=16, labelpad=7)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax1.dist = 9

        if lengths is None or sum(lengths) == 0:
            # ax1
            # fit spline
            lc = gradient_color_change_magic(x, y, z)
            ax1.add_collection(lc)

            if x2 is not None and y2 is not None and z2 is not None:
                lc2 = gradient_color_change_magic(x2, y2, z2)
                ax1.add_collection(lc2)

                x = np.concatenate((x, x2))
                y = np.concatenate((y, y2))
                z = np.concatenate((z, z2))

        elif lengths is not None:
            for i in range(len(lengths)):
                if i == 0:
                    xseg = x[0:lengths[i]]
                    yseg = y[0:lengths[i]]
                    zseg = z[0:lengths[i]]

                else:
                    start_index = sum(lengths[:i])
                    end_index = sum(lengths[:(i+1)])
                    xseg = x[start_index:end_index]
                    yseg = y[start_index:end_index]
                    zseg = z[start_index:end_index]

                lc = gradient_color_change_magic(xseg, yseg, zseg)
                ax1.add_collection(lc)

    # Setting axis limits
    if same_axis:
        max_ = max(x.max(), y.max(), z.max())
        min_ = min(x.min(), y.min(), z.min())
        range_ = max_ - min_
        ax0.set_xlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
        ax0.set_ylim([min_ - 0.1 * range_, max_ + 0.1 * range_])
        if z is not None:
            ax1.set_xlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
            ax1.set_ylim([min_ - 0.1 * range_, max_ + 0.1 * range_])
            ax1.set_zlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
    else:
        xrange_ = x.max() - x.min()
        yrange_ = y.max() - y.min()
        zrange_ = z.max() - z.min()
        ax0.set_xlim([x.min() - 0.1 * xrange_, x.max() + 0.1 * xrange_])
        ax0.set_ylim([y.min() - 0.1 * yrange_, y.max() + 0.1 * yrange_])
        if z is not None:
            ax1.set_xlim([x.min() - 0.1 * xrange_, x.max() + 0.1 * xrange_])
            ax1.set_ylim([y.min() - 0.1 * yrange_, y.max() + 0.1 * yrange_])
            ax1.set_zlim([z.min() - 0.1 * zrange_, z.max() + 0.1 * zrange_])

    if points_to_circle is not None:
        for i in points_to_circle:
            ax0.scatter(x[i], y[i], edgecolors='k', facecolors='none', s=50, zorder=5)
            if z is not None:
                ax1.scatter(x[i], y[i], z[i], edgecolors='k', facecolors='none', s=50, zorder=5)

    fig.tight_layout(pad=5)
    fig.subplots_adjust(top=0.88, wspace=0.2)

    plt.show()
    plt.savefig(os.path.join(output_directory, imgname + ".pdf"))
    plt.clf()


def colored_line_plot_projected_data(x, y, new_data_x, new_data_y=None, z=None, new_data_z=None, x2=None, y2=None,
                                     z2=None, imgname=None, lengths=None, new_data=None, output_directory=None,
                                     points_to_circle=None, points_to_circle_new_data=None, same_axis=False):

    # Create figure
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[0])
    ax0.grid(True)

    ax0.set_xlabel('PC1', fontsize=16)
    ax0.set_ylabel('PC2', fontsize=16)
    ax0.tick_params(axis='both', labelsize=12)

    if lengths is None:
        lc = gradient_color_change_magic(x, y)
        ax0.add_collection(lc)

        if x2 is not None and y2 is not None:
            lc2 = gradient_color_change_magic(x2, y2)
            ax0.add_collection(lc2)

            x = np.concatenate((x, x2))
            y = np.concatenate((y, y2))

        # plotting
        if same_axis:
            max_ = max(x_i.max(), y_i.max())
            min_ = min(x_i.min(), y_i.min())
            range_ = max_ - min_
            ax0.set_xlim([min_-0.1*range_, max_+0.1*range_])
            ax0.set_ylim([min_-0.1*range_, max_+0.1*range_])
        else:
            xrange_ = x_i.max() - x_i.min()
            yrange_ = y_i.max() - y_i.min()
            ax0.set_xlim([x_i.min()-0.1*xrange_, x_i.max()+0.1*xrange_])
            ax0.set_ylim([y_i.min()-0.1*yrange_, y_i.max()+0.1*yrange_])

    elif lengths is not None:
        for i in range(len(lengths)):
            if i == 0:
                xseg = x[0:lengths[i]]
                yseg = y[0:lengths[i]]

                # ax0
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
                x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, array=x_arr, cmap='viridis',
                                    norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax0.add_collection(lc)

            else:
                start_index = sum(lengths[:i])
                end_index = sum(lengths[:(i+1)])
                # print("start_index = %s" % start_index)
                # print("end_index = %s" % end_index)
                xseg = x[start_index:end_index]
                yseg = y[start_index:end_index]

                # ax0
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
                x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, array=x_arr, cmap='viridis',
                                    norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax0.add_collection(lc)

        # Setting axis limits
        if same_axis:
            max_ = max(x.max(), y.max())
            min_ = min(x.min(), y.min())
            range_ = max_ - min_
            ax0.set_xlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
            ax0.set_ylim([min_ - 0.1 * range_, max_ + 0.1 * range_])
        else:
            xrange_ = x.max() - x.min()
            yrange_ = y.max() - y.min()
            ax0.set_xlim([x.min() - 0.1 * xrange_, x.max() + 0.1 * xrange_])
            ax0.set_ylim([y.min() - 0.1 * yrange_, y.max() + 0.1 * yrange_])

    # Adding new_data, if it exists
    if new_data is not None:
        tck_new, u_new = interpolate.splprep([new_data_x, new_data_y], k=1, s=0.0)
        x_i_new, y_i_new = interpolate.splev(np.linspace(0, 1, 1000), tck_new)

        # Gradient color change magic
        z_new = np.linspace(0.0, 1.0, x_i_new.shape[0])
        points_new = np.array([x_i_new, y_i_new]).T.reshape(-1, 1, 2)
        segments_new = np.concatenate([points_new[:-1], points_new[1:]], axis=1)
        lc_new = LineCollection(segments_new, array=z_new, cmap='viridis', norm=plt.Normalize(0.0, 1.0),
                            linestyle="solid", linewidth=2.5)
        lca_new = LineCollection(segments_new, color='black', linestyle="solid", linewidth=5.0)
        ax0.add_collection(lca_new)
        ax0.add_collection(lc_new)

        x_i_new = np.concatenate((x_i, x_i_new))
        y_i_new = np.concatenate((y_i, y_i_new))

        xrange_ = x_i_new.max() - x_i_new.min()
        yrange_ = y_i_new.max() - y_i_new.min()
        ax0.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
        ax0.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])
        # ax0.set_xlim(-40, 100)
        # ax0.set_ylim(-20, 20)
        ax0.ticklabel_format(style='sci', scilimits=(-3, 3))

        if points_to_circle_new_data is not None:
            for i in points_to_circle_new_data:
                ax0.scatter(new_data[0][i], new_data[1][i], edgecolors='k', facecolors='none', s=200, zorder=5, linewidth=1.5)

    if z is not None:
        # Do 3D plot

        ax1 = fig.add_subplot(gs[1], projection='3d')
        # View for SN2 figure
        # ax1.view_init(elev=30., azim=200)

        ax1.set_xlabel('PC1', fontsize=16, labelpad=9)
        ax1.set_ylabel('PC2', fontsize=16, labelpad=7)
        ax1.set_zlabel('PC3', fontsize=16, labelpad=7)
        ax1.tick_params(axis='both', labelsize=12)#, pad=5)
        ax1.ticklabel_format(style='sci', scilimits=(-3, 3))
        # ax1.set_title('Top Three Principal Components', fontsize=18, fontstyle='italic')
        ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax1.dist = 9
        # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if points_to_circle is not None:
        for i in points_to_circle:
            ax0.scatter(x[i], y[i], edgecolors='k', facecolors='none', s=50, zorder=5)
            if z is not None:
                ax1.scatter(x[i], y[i], z[i], edgecolors='k', facecolors='none', s=50, zorder=5)

    if lengths is None or sum(lengths) == 0:
        lc = gradient_color_change_magic(x, y, z)
        ax1.add_collection(lc)

        if x2 is not None and y2 is not None and z2 is not None:
            tck, u = interpolate.splprep([x2, y2, z2], k=1, s=0.0)
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
                zseg = z[0:lengths[i]]

                # ax1
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg, zseg], k=1, s=0.0)
                x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax1.add_collection(lc)

            else:
                start_index = sum(lengths[:i])
                end_index = sum(lengths[:(i+1)])
                # print("start_index = %s" % start_index)
                # print("end_index = %s" % end_index)
                xseg = x[start_index:end_index]
                yseg = y[start_index:end_index]
                zseg = z[start_index:end_index]

                # ax1
                # fit spline
                tck, u = interpolate.splprep([xseg, yseg, zseg], k=1, s=0.0)
                x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                x_arr = np.linspace(0.0, 1.0, x_i.shape[0])
                points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=x_arr, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                ax1.add_collection(lc)

        # Setting axis limits
        if same_axis:
            max_ = max(x.max(), y.max(), z.max())
            min_ = min(x.min(), y.min(), z.min())
            range_ = max_ - min_
            ax1.set_xlim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_ylim([min_-0.1*range_, max_+0.1*range_])
            ax1.set_zlim([min_-0.1*range_, max_+0.1*range_])
        else:
            xrange_ = x.max() - x.min()
            yrange_ = y.max() - y.min()
            zrange_ = z.max() - z.min()
            ax1.set_xlim([x.min()-0.1*xrange_, x.max()+0.1*xrange_])
            ax1.set_ylim([y.min()-0.1*yrange_, y.max()+0.1*yrange_])
            ax1.set_zlim([z.min() - 0.1 * zrange_, z.max() + 0.1 * zrange_])

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

        # ax1.set_xlim(-40, 100)
        # ax1.set_ylim(-20, 20)
        # ax1.set_zlim(-20, 30)

        if points_to_circle_new_data is not None:
            for i in points_to_circle_new_data:
                ax1.scatter(new_data[0][i], new_data[1][i], new_data[2][i], edgecolors='k', facecolors='none', s=200,
                            zorder=5, linewidth=1.5)


    fig.tight_layout(pad=5)
    fig.subplots_adjust(top=0.88, wspace=0.2)

    if imgname is None:
        plt.show()

    else:
        # plt.savefig(output_directory + "/" + imgname + ".png", dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(output_directory, imgname + ".pdf"))


def colored_line_and_scatter_plot(x, y=None, y1=None, x2=None, y2=None, y12=None, imgname=None, same_axis=False, lengths=None, new_data=None,
              output_directory=None, points_to_circle=None, mark_first_last=False):
    """
    Create a 2D plot or 1D if y == None
    """

    if y is None and y1 is None:
        plt.figure(figsize=(8, 4))
        # Do 1D plot
        plt.plot(range(len(x)), x)
    else:
        # Do 2D plot

        # Create figure
        fig = plt.figure(figsize=(10, 5))
        # gs = GridSpec(2, 1, height_ratios=[2, 1])
        gs = GridSpec(1, 2)
        ax0 = fig.add_subplot(gs[0])
        ax0.grid(True)

        ax0.set_xlabel('PC1', fontsize=16)
        ax0.set_ylabel('PC2', fontsize=16)
        ax0.tick_params(axis='both', labelsize=12)
        ax0.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # ax1 = plt.subplot(gs[1])

        if lengths is None:
            # ax0
            # fit spline
            tck, u = interpolate.splprep([x, y], k=1, s=0.0)
            x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

            # Gradient color change magic
            z = np.linspace(0.0, 1.0, x_i.shape[0])
            points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8, linewidth=2)
            ax0.add_collection(lc)
            time = list(range(len(x)))
            ax0.scatter(x, y, s=50, c=time, cmap='viridis', zorder=10, alpha=0.6)

            if x2 is not None and y2 is not None:
                tck2, u2 = interpolate.splprep([x2, y2], k=1, s=0.0)
                x_i2, y_i2 = interpolate.splev(np.linspace(0, 1, 1000), tck2)

                # Gradient color change magic
                z = np.linspace(0.0, 1.0, x_i2.shape[0])
                points2 = np.array([x_i2, y_i2]).T.reshape(-1, 1, 2)
                segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
                lc2 = LineCollection(segments2, array=z, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8,
                                    linewidth=2)
                ax0.add_collection(lc2)
                time = list(range(len(x2)))
                ax0.scatter(x2, y2, s=50, c=time, cmap='viridis', zorder=10, alpha=0.6)

                x_i = np.concatenate((x_i, x_i2))
                y_i = np.concatenate((y_i, y_i2))

            # plotting
            if same_axis:
                max_ = max(x_i.max(), y_i.max())
                min_ = min(x_i.min(), y_i.min())
                range_ = max_ - min_
                ax0.set_xlim([min_-0.1*range_, max_+0.1*range_])
                ax0.set_ylim([min_-0.1*range_, max_+0.1*range_])
            else:
                xrange_ = x_i.max() - x_i.min()
                yrange_ = y_i.max() - y_i.min()
                ax0.set_xlim([x_i.min()-0.1*xrange_, x_i.max()+0.1*xrange_])
                ax0.set_ylim([y_i.min()-0.1*yrange_, y_i.max()+0.1*yrange_])

            # time plots
            # ax1.plot(range(len(x)), x)
            # ax1.plot(range(len(y)), y)

        elif lengths is not None:
            for i in range(len(lengths)):
                if i == 0:
                    xseg = x[0:lengths[i]]
                    yseg = y[0:lengths[i]]

                    # ax0
                    # fit spline
                    tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                    # Gradient color change magic
                    z = np.linspace(0.0, 1.0, x_i.shape[0])
                    points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, array=z, cmap='viridis',
                                        norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                    ax0.add_collection(lc)
                    time = list(range(len(xseg)))
                    ax0.scatter(xseg, yseg, s=50, c=time, cmap='viridis', zorder=10)
                    if mark_first_last:
                        ax0.scatter(xseg.iloc[[0]], yseg.iloc[[0]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
                        ax0.scatter(xseg.iloc[[-1]], yseg.iloc[[-1]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
                        ax0.annotate("%s" % (i+1), (xseg.iloc[[-1]], yseg.iloc[[-1]]),
                                     ha='center', va='center', zorder=100)
                else:
                    start_index = sum(lengths[:i])
                    end_index = sum(lengths[:(i+1)])
                    # print("start_index = %s" % start_index)
                    # print("end_index = %s" % end_index)
                    xseg = x[start_index:end_index]
                    yseg = y[start_index:end_index]

                    # ax0
                    # fit spline
                    tck, u = interpolate.splprep([xseg, yseg], k=1, s=0.0)
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, 1000), tck)

                    # Gradient color change magic
                    z = np.linspace(0.0, 1.0, x_i.shape[0])
                    points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, array=z, cmap='viridis',
                                        norm=plt.Normalize(0.0, 1.0), alpha=0.8)
                    ax0.add_collection(lc)
                    time = list(range(len(xseg)))
                    ax0.scatter(xseg, yseg, s=50, c=time, cmap='viridis', zorder=10)
                    if mark_first_last:
                        ax0.scatter(xseg.iloc[[0]], yseg.iloc[[0]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
                        ax0.scatter(xseg.iloc[[-1]], yseg.iloc[[-1]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
                        ax0.annotate("%s" % (i+1), (xseg.iloc[[-1]], yseg.iloc[[-1]]),
                                     ha='center', va='center', zorder=10)

            # Setting axis limits
            if same_axis:
                max_ = max(x.max(), y.max())
                min_ = min(x.min(), y.min())
                range_ = max_ - min_
                ax0.set_xlim([min_ - 0.1 * range_, max_ + 0.1 * range_])
                ax0.set_ylim([min_ - 0.1 * range_, max_ + 0.1 * range_])
            else:
                xrange_ = x.max() - x.min()
                yrange_ = y.max() - y.min()
                ax0.set_xlim([x.min() - 0.1 * xrange_, x.max() + 0.1 * xrange_])
                ax0.set_ylim([y.min() - 0.1 * yrange_, y.max() + 0.1 * yrange_])

        # Adding new_data, if it exists
        if new_data is not None:

            tck_new, u_new = interpolate.splprep([new_data[0], new_data[1]], k=1, s=0.0)
            x_i_new, y_i_new = interpolate.splev(np.linspace(0, 1, 1000), tck_new)

            # Gradient color change magic
            z_new = np.linspace(0.0, 1.0, x_i_new.shape[0])
            points_new = np.array([x_i_new, y_i_new]).T.reshape(-1, 1, 2)
            segments_new = np.concatenate([points_new[:-1], points_new[1:]], axis=1)
            lc_new = LineCollection(segments_new, array=z_new, cmap='viridis', norm=plt.Normalize(0.0, 1.0),
                                linestyle="solid", linewidth=2.5)
            lca_new = LineCollection(segments_new, color='black', linestyle="solid", linewidth=6.0)
            ax0.add_collection(lca_new)
            ax0.add_collection(lc_new)
            time = list(range(len(new_data[0])))
            ax0.scatter(new_data[0], new_data[1], s=5, c=time, cmap='viridis', zorder=11)

            x_i_new = np.concatenate((x_i, x_i_new))
            y_i_new = np.concatenate((y_i, y_i_new))

            xrange_ = x_i_new.max() - x_i_new.min()
            yrange_ = y_i_new.max() - y_i_new.min()
            ax0.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
            ax0.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])

    if y1 is not None:
        # Do 3D plot

        ax1 = fig.add_subplot(gs[1], projection='3d')
        # View for SN2 figure
        # ax1.view_init(elev=50., azim=200)

        ax1.set_xlabel('PC1', fontsize=16, labelpad=9)
        ax1.set_ylabel('PC2', fontsize=16, labelpad=9)
        ax1.set_zlabel('PC3', fontsize=16, labelpad=9)
        ax1.tick_params(axis='both', labelsize=12)#, pad=5)
        ax1.ticklabel_format(style='sci', scilimits=(-3, 3))
        # ax1.set_title('Top Three Principal Components', fontsize=18, fontstyle='italic')
        ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax1.dist = 9
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

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
            ax1.scatter(x, y, y1, s=50, c=list(range(len(x))), cmap='viridis', alpha=0.6)

            if x2 is not None and y2 is not None and y12 is not None:
                tck, u = interpolate.splprep([x2, y2, y12], k=1, s=0.0)
                x_i2, y_i2, z_i2 = interpolate.splev(np.linspace(0, 1, 1000), tck)

                # Gradient color change magic
                z2 = np.linspace(0.0, 1.0, x_i2.shape[0])
                points = np.array([x_i2, y_i2, z_i2]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc2 = Line3DCollection(segments, array=z2, cmap='viridis', norm=plt.Normalize(0.0, 1.0), alpha=0.8,
                                      linewidth=2, zorder=2)
                ax1.add_collection(lc2)
                ax1.scatter(x2, y2, y12, s=50, c=list(range(len(x2))), cmap='viridis', zorder=1)

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
                    ax1.scatter(xseg, yseg, zseg, s=50, c=list(range(len(xseg))), cmap='viridis')
                    if mark_first_last:
                        ax1.scatter(xseg.iloc[[0]], yseg.iloc[[0]], zseg.iloc[[0]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
                        ax1.scatter(xseg.iloc[[-1]], yseg.iloc[[-1]], zseg.iloc[[-1]], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)

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
                    ax1.scatter(xseg, yseg, zseg, s=50, c=list(range(len(xseg))), cmap='viridis')
                    if mark_first_last:
                        ax1.scatter(xseg.iloc[[0]], yseg.iloc[[0]], zseg.iloc[[0]], edgecolors='k', linewidth=1.5,
                                    facecolors='none', s=200, zorder=10)
                        ax1.scatter(xseg.iloc[[-1]], yseg.iloc[[-1]], zseg.iloc[[-1]], edgecolors='k', linewidth=1.5,
                                    facecolors='none', s=200, zorder=10)

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
            lca_new = Line3DCollection(segments_new, color='black', linestyle="solid", linewidth=6.0)
            ax1.add_collection(lca_new)
            ax1.add_collection(lc_new)
            ax1.scatter(new_data[0], new_data[1], new_data[2], s=5, c=list(range(len(new_data[0]))), cmap='viridis')

            x_i_new = np.concatenate((x_i, x_i_new))
            y_i_new = np.concatenate((y_i, y_i_new))
            z_i_new = np.concatenate((z_i, z_i_new))

            xrange_ = x_i_new.max() - x_i_new.min()
            yrange_ = y_i_new.max() - y_i_new.min()
            zrange_ = z_i_new.max() - z_i_new.min()
            ax1.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
            ax1.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])
            ax1.set_zlim([z_i_new.min() - 0.1 * zrange_, z_i_new.max() + 0.1 * zrange_])

    if points_to_circle is not None:
        for i in points_to_circle:
            ax0.scatter(x[i], y[i], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)
            ax1.scatter(x[i], y[i], y1[i], edgecolors='k', linewidth=1.5, facecolors='none', s=200, zorder=10)

    fig.tight_layout(pad=5)
    fig.subplots_adjust(top=0.88, wspace=0.2)

    if imgname is None:
        plt.show()

    else:
        # plt.savefig(output_directory + "/" + imgname + ".png", dpi=1200, bbox_inches='tight')
        # plt.savefig(output_directory + "/" + imgname + ".eps")
        plt.show()
        plt.savefig(os.path.join(output_directory, imgname + ".pdf"))
        plt.close()


def colored_scatter_plot(x, y, z, directory=None, points_to_circle=None, imgname=None):

    time = list(range(len(x)))

    if directory is None:
        directory = "%s_eps_files" % imgname
        if not os.path.exists(directory):
            os.makedirs(directory)

    x = list(x)
    y = list(y)
    z = list(z)

    # Create figure
    fig0 = plt.figure(figsize=(6, 5))
    ax0 = fig0.add_subplot(1, 1, 1)
    # ax0.grid(True)

    ax0.scatter(x, y, s=100, c=time, cmap='viridis', edgecolors='k')
    ax0.set_xlabel("PC1", fontsize=16)
    ax0.set_ylabel("PC2", fontsize=16)

    ax0.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
    ax0.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
    ax0.tick_params(labelsize=14, pad=10)

    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter(x, y, z, s=100, c=time, cmap='viridis', edgecolors='k')
    ax1.set_xlabel("PC1", fontsize=16, labelpad=20)
    ax1.set_ylabel("PC2", fontsize=16, labelpad=20)
    ax1.set_zlabel("PC3", fontsize=16, labelpad=20)
    ax1.ticklabel_format(style='sci', scilimits=(-4,4))

    ax1.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
    ax1.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
    ax1.tick_params(labelsize=14, pad=10)

    # ax1.view_init(elev=20., azim=60)
    ax1.grid(False)

    if points_to_circle is not None:
        # ax0.scatter(x[point], y[point], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=2)
        # ax1.scatter(x[point], y[point], z[point], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=2)
        for i in points_to_circle:
            ax0.scatter(x[i], y[i], edgecolors='k', facecolors='none', s=400, linewidth=2, zorder=5)
            ax1.scatter(x[i], y[i], z[i], edgecolors='k', facecolors='none', s=400, linewidth=2, zorder=5)

        filename = "%s_scatter.eps" % imgname

        fig0.savefig(directory + "/" + "2D_" + filename, format='eps', bbox_inches='tight')
        fig1.savefig(directory + "/" + "3D_" + filename, format='eps', bbox_inches='tight', pad_inches=0.5)

    else:
        filename = "%s_scatter.eps" % imgname
        fig0.savefig(directory + "/" + "2D_" + filename, format='eps', bbox_inches='tight')
        fig1.savefig(directory + "/" + "3D_" + filename, format='eps', bbox_inches='tight', pad_inches=0.5)

        plt.show()


def plot_irc(path, imgname=None, output_directory=None, points_to_circle=None, gif_points_to_circle=None):

    data = pd.read_csv(path, header=None, delim_whitespace=True)

    data[2] = (data[1] - data[1][0]) * 627.51

    fig = plt.figure(figsize=(10, 5))

    x = data[0]
    y = data[2]

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=100, c=x, cmap='viridis', edgecolors='k')
    ax.set_xlabel("Intrinsic Reaction Coordinate (Bohr$\sqrt{amu}$)", fontsize=16)
    ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=16)

    ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
    ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
    plt.tick_params(labelsize=14)

    if points_to_circle is not None:
        for i in points_to_circle:
            ax.scatter(x[i], y[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=2)

    if output_directory and imgname:
        plt.savefig(os.path.join(output_directory, imgname), format='eps')


def plot_irc_for_gif(path, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled. To stitch them into a gif, use ImageMagick:
    convert -delay 8 $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param path: path to IRC energies (txt or csv file)
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.read_csv(path, header=None, delim_whitespace=True)

    data[2] = (data[1] - data[1][0]) * 627.51

    x = data[0]
    y = data[2]

    for i in gif_points_to_circle:

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(1, 1, 1)
        lc = gradient_color_change_magic(x, y)
        ax.add_collection(lc)
        ax.scatter(x, y, s=100, c=x, cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("Intrinsic Reaction Coordinate (Bohr$\sqrt{amu}$)", fontsize=16)
        ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=16)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        plt.tick_params(labelsize=14)

        ax.scatter(x[i], y[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        plt.savefig(os.path.join(output_directory, imgname + "_scatterline_%s.pdf" % i), format='pdf')

        plt.close(fig)

    plt.close('all')

def plot_pcs_for_gif_2D(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]

    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))

        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)

        lc = gradient_color_change_magic(x, y)
        ax.add_collection(lc)
        ax.scatter(x, y, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.tick_params(labelsize=14)
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        plt.savefig(os.path.join(output_directory, imgname + "_2PCs_scatterline_%s.pdf" % i), bbox_inches='tight',
                    format='pdf')

        plt.close(fig)

    plt.close('all')


def plot_pcs_for_gif_3D(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]
    z = data[2]

    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=25., azim=55)

        lc = gradient_color_change_magic(x, y, z)
        ax.add_collection(lc)
        ax.scatter(x, y, z, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16, labelpad=10)
        ax.set_ylabel("PC2", fontsize=16, labelpad=10)
        ax.set_zlabel("PC3", fontsize=16, labelpad=15)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.set_zlim(min(z) - 0.1 * max(z), 1.1 * max(z))
        ax.tick_params(labelsize=14, axis='z', pad=10)
        ax.tick_params(labelsize=14, axis='x')
        ax.tick_params(labelsize=14, axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], z[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        # Set viewing "distance" to plot, so axes aren't cut off in final images
        ax.dist = 9
        plt.savefig(os.path.join(output_directory, imgname + "_3PCs_scatterline_%s.pdf" % i), format='pdf')

        plt.close(fig)

    plt.close('all')


def plot_pcs_for_gif_3D_rotating(pca_matrix, gif_points_to_circle, imgname=None, output_directory=None):
    """
    Will generate pictures of a plot with individual points circled (number of pictures = number of points to circle).
    To stitch them into a gif, use ImageMagick:
    convert -delay 8 {-density 400} $(for i in $(seq 1 `ls *.pdf | wc -l`); do echo imgname_${i}.pdf; done) n-loop 0 gif-name.gif
    :param pca_matrix: numpy array, matrix of PCs after conducting PCA
    :param gif_points_to_circle: list of points (ints) to circle
    :param imgname: name of group of images (e.g., the system name)
    :param output_directory: where to put output images
    :return: None
    """

    data = pd.DataFrame(pca_matrix)

    x = data[0]
    y = data[1]
    z = data[2]

    rot = 0
    for i in gif_points_to_circle:
        fig = plt.figure(figsize=(6, 5))
        rot += 360/len(x)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=25., azim=rot)

        lc = gradient_color_change_magic(x, y, z)
        ax.add_collection(lc)
        ax.scatter(x, y, z, s=100, c=list(range(len(x))), cmap='viridis', edgecolors='k', zorder=2)
        ax.set_xlabel("PC1", fontsize=16, labelpad=10)
        ax.set_ylabel("PC2", fontsize=16, labelpad=10)
        ax.set_zlabel("PC3", fontsize=16, labelpad=20)

        ax.set_xlim(min(x) - 0.1 * max(x), 1.1 * max(x))
        ax.set_ylim(min(y) - 0.1 * max(y), 1.1 * max(y))
        ax.set_zlim(min(z) - 0.1 * max(z), 1.1 * max(z))
        ax.tick_params(labelsize=14, axis='z', pad=10)
        ax.tick_params(labelsize=14, axis='x')
        ax.tick_params(labelsize=14, axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.scatter(x[i], y[i], z[i], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=-1)
        # Set viewing "distance" to plot, so axes aren't cut off in final images
        ax.dist = 11
        plt.savefig(os.path.join(output_directory, imgname + "_3PCs_scatterline_rotating_%s.pdf" % i), format='pdf')

        plt.close(fig)

    plt.close('all')


def plot_prop_of_var(values, system_name=None, output_directory=None):
    """
    Plot the proportion of variance using the eigenvalues of the covariance matrix of the input data.
    :param values: array or list, eigenvalues of the covariance matrix (pca.explained_variance_)
    :param system_name: str, name of the system to use as file name
    :param output_directory: str, directory in which to put output plot
    :return: None
    """
    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    normalized_values = values / np.sum(values)
    x = range(len(values))

    ax.scatter(x, normalized_values, c='k')

    ax.set_xlabel("Principal Component", fontsize=16)
    ax.set_ylabel("Proportion of Variance", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_ylim(-0.1, 1.1)

    cumulative = np.cumsum(normalized_values)

    ax1.scatter(x, cumulative)
    ax1.set_xlabel("Principal Component", fontsize=16)
    ax1.set_ylabel("Cumulative Prop. of Var.", fontsize=16)
    ax1.tick_params(labelsize=14)
    ax1.set_ylim(-0.1, 1.1)

    fig.tight_layout()

    if system_name is None or output_directory is None:
        plt.show()

    else:
        fig.savefig(os.path.join(output_directory, '%s_proportion_of_variance.png' % system_name), dpi=600)