import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
file = '../examples/malonaldehyde/malonaldehyde_IRC.xyz'
stereo_atoms_malon = [4, 7, 1, 8]
points_to_circle = [0, 12, 24]


# DISTANCES INPUT, NOT MASS-WEIGHTED. To mass-weight coordinates, add "MW=True" to function call.
system_name, output_directory_D, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths_D, \
aligned_original_coords = dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_malon, input_type="Distances")

# Plot results
D_pca_df = pd.DataFrame(D_pca)
colored_line_and_scatter_plot(D_pca_df[0], D_pca_df[1], D_pca_df[2], output_directory=output_directory_D,
          imgname=(system_name + "_Distances_noMW_scatterline"), points_to_circle=points_to_circle)




