import pandas as pd
import dimensionality_reduction_functions as dim_red
import numpy as np
from plotting_functions import colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

#################################################### EXAMPLE 2: SN2 ####################################################
# Input files and atom numbers surrounding stereogenic center
file = '../examples/SN2/SN2_IRC.xyz'
stereo_atoms_SN2 = [1, 4, 5, 2]
new_file = '../examples/SN2/SN2_traj1.xyz'

# DISTANCES INPUT. To mass-weight coordinates, add "MW=True" to function call.
system_name, output_directory_D, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths, aligned_original_coords = \
   dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Transforming new data into RD space
new_data_D_df = dim_red.transform_new_data(new_file, output_directory_D + "/new_data", ndim, D_pca_fit,
                                           D_pca_components, D_mean, aligned_original_coords,
                                           stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Plot Results
D_pca_df = pd.DataFrame(D_pca)
points_to_circle = [0, 15, np.argmin(np.array(D_pca_df[1])), 103]

# Original Data
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, output_directory=output_directory_D,
          imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)

# New Data
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_D_df,
          output_directory=output_directory_D + "/new_data", imgname=(system_name + "_Distances_noMW_scatterline"),
                              points_to_circle=points_to_circle)



