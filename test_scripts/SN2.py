import pandas as pd
import dimensionality_reduction_functions as dim_red
import numpy as np
from plotting_functions import colored_line_plot, colored_line_and_scatter_plot, plot_irc

# Number of PCA components
ndim = 3

#################################################### EXAMPLE 2: SN2 ####################################################
# Input files and atom numbers surrounding stereogenic center
file = '../examples/SN2/SN2_IRC.xyz'
stereo_atoms_SN2 = [1, 4, 5, 2]
new_file = '../examples/SN2/SN2_traj1.xyz'

# DISTANCES INPUT. To mass-weight coordinates, add "MW=True" to function call.
system_name1, output_directory_D, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1 = \
   dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Transforming new data into RD space
new_data_D_df = dim_red.transform_new_data(new_file, output_directory_D + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                           stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Plot results
D_pca_df = pd.DataFrame(D_pca)
points_to_circle = [0, 103 - 15, 103 - np.argmin(D_pca_df[1]), 103]
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, output_directory=output_directory_D,
          imgname=(system_name1 + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)
colored_line_and_scatter_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_D_df,
          output_directory=output_directory_D + "/new_data", imgname=(system_name1 + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)
plot_irc('/Users/ec18006/Documents/CHAMPS/Dimensionality_reduction/xyz_pdb_files/examples/SN2/',
         'SN2_IRC', output_directory_D, points_to_circle=points_to_circle)

# # CARTESIANS INPUT. To mass-weight coordinates, add "MW=True" to function call.
# system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, traj_lengths = \
#     dim_red.pathreducer(file, ndim, input_type="Cartesians", MW=True)
#
# # Transforming new data into RD space
# new_data_df = dim_red.transform_new_data(new_file, direc + "/new_data", ndim, coords_pca_fit, coords_comps, coords_mean,
#                                          input_type="Cartesians", MW=True)
#
# # Plot results
# coords_pca_df = pd.DataFrame(coords_pca)
# colored_line_plot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, output_directory=direc,
#           imgname=(system_name + "_Cartesians_MW"))
# colored_line_plot(coords_pca_df[0], y=coords_pca_df[1], y1=coords_pca_df[2], same_axis=False, new_data=new_data_df,
#           output_directory=direc + "/new_data", imgname=(system_name + "_Cartesians_MW"))


