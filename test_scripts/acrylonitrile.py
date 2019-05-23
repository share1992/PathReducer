import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_plot, colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 3: ACRYLONITRILE ###############################################
# Inputs
file = '../examples/acrylonitrile/acrylonitrile_scan.xyz'
stereo_atoms_acryl = [1, 2, 3, 4]


# # DISTANCES INPUT
# system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1 = \
#    dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_acryl, input_type="Distances", MW=False)
#
# # Plot results
# D_pca_df = pd.DataFrame(D_pca)
# colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, output_directory=direc1,
#           imgname=(system_name1 + "_Distances_noMW_scatterline"))


# CARTESIANS INPUT
system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, traj_lengths = \
    dim_red.pathreducer(file, ndim, input_type="Cartesians")

# Plot results
coords_pca_df = pd.DataFrame(coords_pca)
colored_line_and_scatter_plot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, output_directory=direc,
          imgname=(system_name + "_Cartesians_noMW_scatterline"))


