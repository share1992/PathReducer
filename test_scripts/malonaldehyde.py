import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_plot, colored_scatter_plot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
file = '../examples/malonaldehyde/malonaldehyde_IRC.xyz'
stereo_atoms_malon = [4, 7, 1, 8]

# CARTESIANS INPUT
traj_lengths, system_name, output_directory, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, = \
    dim_red.dr_routine(file, ndim, input_type="Cartesians")

# DISTANCES INPUT
traj_lengths, system_name, output_directory, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_malon, input_type="Distances")


# Plotting PCs in 2D and 3D
points_to_circle = [0, 12, 24]

# CARTESIANS INPUT
coords_pca_df = pd.DataFrame(coords_pca)
colored_line_plot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, output_directory=output_directory,
          imgname=system_name, points_to_circle=points_to_circle)

# DISTANCES INPUT
D_pca_df = pd.DataFrame(D_pca)
colored_line_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, output_directory=output_directory,
          imgname=(system_name + "_D"), points_to_circle=points_to_circle)

colored_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], output_directory, points_to_circle=points_to_circle,
                     imgname=(system_name + "scatter_D"), )
