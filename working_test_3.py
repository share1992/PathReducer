import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot
from scatter_plot import colorplot_scatter

################################################### Input Variables  ###################################################

file = 'robins_files/Trimmed.xyz'

# Number of PCA components
ndim = 3

# stereo_atoms = [5, 6, 7, 8]

############################################### COORDINATES input to PCA ###############################################

traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
    dim_red.dr_routine(file, ndim, input_type="Coordinates")

# Plot trajectory in 2D and 3D
coords_pca_df = pd.DataFrame(coords_pca)
colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
          output_directory=direc, imgname=system_name)

################################################ DISTANCES input to PCA ################################################

traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, input_type="Distances")

# Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
D_pca_df = pd.DataFrame(D_pca)
colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc1,
          imgname=(system_name1 + "_D"))

# Print coefficients of distances that make up principal components to text files, put into output directory
dim_red.print_distance_coeffs_to_files(direc1, ndim, system_name1, D_pca_components, 179)
dim_red.print_distance_coeffs_to_files_weighted(direc1, ndim, system_name1, D_pca_components, D_values, 179)

