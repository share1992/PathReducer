import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 3: ACRYLONITRILE ###############################################
# Inputs
file = '../examples/acrylonitrile/acrylonitrile_scan.xyz'
stereo_atoms_acryl = [1, 2, 3, 4]

# DISTANCES INPUT
system_name, output_directory, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths, \
aligned_original_coords = dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_acryl, input_type="Distances", MW=False)

# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, output_directory=output_directory,
          imgname=(system_name + "_Distances_noMW_scatterline"))




