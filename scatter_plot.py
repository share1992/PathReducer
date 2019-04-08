import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def colorplot_scatter(x, y, z, time, point=None, name=None):

    if name is not None:
        directory = "%s_eps_files" % name
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

    if point is not None:
        ax0.scatter(x[point], y[point], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=2)
        ax1.scatter(x[point], y[point], z[point], s=400, facecolors='none', edgecolors="black", linewidth=2, zorder=2)

        filename = "%s_scatter_%s.eps" % (name, point)

        # fig0.savefig(directory + "/" + "2D" + filename, format='eps', bbox_inches='tight')
        fig1.savefig(directory + "/" + "3D_new_angle" + filename, format='eps', bbox_inches='tight', pad_inches=0.5)

    else:
        filename = "%s_scatter.eps" % name
        # fig0.savefig(directory + "/" + "2D" + filename, format='eps', bbox_inches='tight')
        fig1.savefig(directory + "/" + "3D_new_angle" + filename, format='eps', bbox_inches='tight', pad_inches=0.5)

        plt.show()

