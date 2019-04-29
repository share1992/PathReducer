import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

# Number of PCA components
ndim = 3

############################################### EXAMPLE 1: MALONALDEHYDE ###############################################
# Input file and list of atom numbers surrounding stereogenic center
file = 'examples/for_paper/malondialdehydeTS_IRC_for_paper.xyz'
stereo_atoms_malon = [4, 7, 1, 8]

# CARTESIANS INPUT
traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
    dim_red.dr_routine(file, ndim, input_type="Cartesians", mass_weighting=True)
# Plot trajectory in 2D and 3D
coords_pca_df = pd.DataFrame(coords_pca)
colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Cartesians",
          output_directory=direc, imgname=system_name)

# DISTANCES INPUT
traj_lengths1, system_name, direc, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_malon, input_type="Distances", mass_weighting=True)
# Plot trajectory in 2D and 3D
D_pca_df = pd.DataFrame(D_pca)
colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc,
          imgname=(system_name + "_D"))
