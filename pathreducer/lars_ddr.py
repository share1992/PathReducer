import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter


def colorplot(x, y=None, y1=None, x2=None, y2=None, y12=None, imgname=None, same_axis=True, input_type=None, lengths=None, new_data=None,
              output_directory=None):
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

        ax0.set_xlabel('P.C. 1', fontsize=16)
        ax0.set_ylabel('P.C. 2', fontsize=16)
        ax0.tick_params(axis='both', labelsize=12)
        # ax0.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax0.set_title('Top Two Principal Components', fontsize=18, fontstyle='italic')

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

            # Malonaldehyde
            # ax0.scatter(x[0], y[0], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.scatter(x[12], y[12], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.scatter(x[24], y[24], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # SN2
            # ax0.scatter(x[0],y[0], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.scatter(x[15],y[15], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.scatter(x[24],y[24], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.scatter(x[103],y[103], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
            # ax0.annotate('A', (x[0]+20, y[0]+10), fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)
            # ax0.annotate('B', (x[20]+20, y[20]+10), fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)
            # ax0.annotate('C', (x[45]+20, y[45]+10), fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)

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
            lca_new = LineCollection(segments_new, color='black', linestyle="solid", linewidth=5.0)
            ax0.add_collection(lca_new)
            ax0.add_collection(lc_new)

            x_i_new = np.concatenate((x_i, x_i_new))
            y_i_new = np.concatenate((y_i, y_i_new))

            xrange_ = x_i_new.max() - x_i_new.min()
            yrange_ = y_i_new.max() - y_i_new.min()
            ax0.set_xlim([x_i_new.min() - 0.1 * xrange_, x_i_new.max() + 0.1 * xrange_])
            ax0.set_ylim([y_i_new.min() - 0.1 * yrange_, y_i_new.max() + 0.1 * yrange_])

    if y1 is not None:
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

    # fig.suptitle('Pathway in Reduced Dimensional Space: %s input' % input_type, fontsize=20)
    # fig.suptitle('%s input' % input_type, fontsize=20)
    fig.tight_layout(pad=5)
    fig.subplots_adjust(top=0.88, wspace=0.02)

    if imgname is None:
        plt.show()

    else:
        # plt.savefig(output_directory + "/" + imgname + ".png", dpi=600, bbox_inches='tight')
        plt.savefig(output_directory + "/" + imgname + ".png", dpi=600)
        plt.clf()

# ax1.scatter(x[0],y[0],y1[0], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
# ax1.scatter(x[20],y[20],y1[20], edgecolors = 'k', facecolors = 'none', s=50, zorder=10)
# ax1.scatter(x[45],y[45],y1[45], edgecolors= 'k', facecolors = 'none', s=50, zorder=10)
# ax1.text(x[0]-40, y[0]+10, y1[0]+10, 'A', fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)
# ax1.text(x[20]+20, y[20]+10, y1[20]+5, 'B', fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)
# ax1.text(x[45]+20, y[45]+10, y1[45]+5, 'C', fontsize=20, fontweight='bold', fontfamily='Arial', zorder=10)
