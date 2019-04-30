import pandas as pd
import dimensionality_reduction_functions as dim_red
import numpy as np
from plotting_functions import colored_line_plot

# Number of PCA components
ndim = 3

#################################################### EXAMPLE 2: SN2 ####################################################
# Input files and atom numbers surrounding stereogenic center
file = '../examples/SN2/SN2_IRC.xyz'
stereo_atoms_SN2 = [1, 4, 5, 2]
new_file = '../examples/SN2/SN2_traj1.xyz'

# CARTESIANS INPUT
system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, traj_lengths = \
    dim_red.pathreducer(file, ndim, input_type="Cartesians")

# Transforming new data into RD space
new_data_df = dim_red.transform_new_data(new_file, direc + "/new_data", ndim, coords_pca_fit, coords_comps, coords_mean,
                                         input_type="Cartesians")

# DISTANCES INPUT
system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1 = \
   dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_SN2, input_type="Distances")

# Transforming new data into RD space
new_data_D_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                           stereo_atoms=stereo_atoms_SN2, input_type="Distances")


# Plotting
coords_pca_df = pd.DataFrame(coords_pca)
D_pca_df = pd.DataFrame(D_pca)
# CARTESIANS INPUT
points_to_circle = [0, 15, np.argmin(D_pca_df[1]), 103]
colored_line_plot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, output_directory=direc,
          imgname=system_name, points_to_circle=points_to_circle)
colored_line_plot(coords_pca_df[0], y=coords_pca_df[1], y1=coords_pca_df[2], same_axis=False, new_data=new_data_df,
          output_directory=direc + "/new_data", imgname=system_name, points_to_circle=points_to_circle)
# DISTANCES INPUT
colored_line_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, output_directory=direc1,
          imgname=(system_name1 + "_D"), points_to_circle=points_to_circle)
colored_line_plot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_df,
          output_directory=direc1 + "/new_data", imgname=(system_name1 + "_D"), points_to_circle=points_to_circle)
