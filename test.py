import pandas as pd
import glob
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot

################################################### Input Variables  ###################################################
# Single files
# file = 'examples/nanotube/NanoTraj.xyz'
# file = 'examples/2-ala-neb-final/2-ala-neb-final.xyz'
# file = 'examples/bifurcations/bifur.xyz'
# file = 'examples/Becca_2QWK/2QWK_attempt_2_traj_positions.xyz'
# file = 'examples/ose_wt/ose_wt_180turnleft_coords.xyz'
# file = 'examples/trypsin.xyz'
# file = 'examples/neuraminidase.xyz'
# file = 'examples/hiv1_protease.xyz'

# Directories
# files = 'examples/ose_wt'
files = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase/n1'

# "New Files" to test transforming trajectories into already generated reduced dimensional space
# new_file = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase/comet-n30-traj6_fixed.xyz'
new_file = 'examples/bifurcations/Pummerer-gas-TS1-IRC.xyz'
# new_file = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase/comet-n2-traj3_fixed.xyz'

# Number of PCA components
ndim = 3

# Atom indexes surrounding a stereogenic center
atom_1 = 1
atom_2 = 2
atom_3 = 3
atom_4 = 4

# Pummerer system atoms surrounding resultant stereogenic center
P_atom_1 = 1
P_atom_2 = 6
P_atom_3 = 27
P_atom_4 = 32


############################################### COORDINATES input to PCA ###############################################
# EXAMPLE 1A: Directory as input
traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full_coords, \
red_coords = dim_red.dr_routine(files, ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, input_type="Coordinates")

# Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
coords_pca_df = pd.DataFrame(coords_pca)
colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
          output_directory=direc, lengths=traj_lengths)

# Transforming new data into generated reduced dimensional space
dim_red.transform_new_data(new_file, direc + "/new_data", ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, coords_pca_fit,
                           coords_comps, coords_mean, coords_pca, traj_lengths, input_type="Coordinates")


# # EXAMPLE 1B: Single file as input
# traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
#     dim_red.dr_routine(file, ndim, atom_1, atom_2, atom_3, atom_4, input_type="Coordinates", filtered_distances=False)
#
# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# coords_pca_df = pd.DataFrame(coords_pca)
# colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
#           output_directory=direc, lengths=traj_lengths)
#
# # Filter top distances. Doesn't seem to be working correctly
# sorted_pc1, sorted_pc2, sorted_pc3 = dim_red.filter_top_distances(system_name, direc, ndim, coords_comps, full, red,
#                                                                   50, 7.0)


##############################################£# DISTANCES input to PCA ####################################£###########
# # EXAMPLE 2A: Directory as input
# traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
#    dim_red.dr_routine(files, ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, input_type="Distances")
#
# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# D_pca_df = pd.DataFrame(D_pca)
# colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", lengths=traj_lengths1)
#
# # Print coefficients of distances that make up principal components to text files, put into output directory
# dim_red.print_distance_coeffs_to_files(direc1, ndim, system_name1, atoms, D_pca_components)


# # EXAMPLE 2B: Single file as input
# traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
#    dim_red.dr_routine(file, ndim, atom_1, atom_2, atom_3, atom_4, input_type="Distances")
#
# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# D_pca_df = pd.DataFrame(D_pca)
# colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances")
#
# # Print coefficients of distances that make up principal components to text files, put into output directory
# dim_red.print_distance_coeffs_to_files(direc1, ndim, system_name1, atoms, D_pca_components)
#
# # Transforming new data into generated reduced dimensional space
# dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, D_pca_fit,
# D_pca_components, D_mean, D_pca, traj_lengths1, input_type="Distances")

