import pandas as pd
import dimensionality_reduction_functions as dim_red
from lars_ddr import colorplot
from scatter_plot import colorplot_scatter
import numpy as np

################################################### Input Variables  ###################################################
## FOR PAPER
# file = 'examples/for_paper/malondialdehydeTS_IRC_for_paper.xyz'
# file = 'examples/for_paper/SN2_all_for_paper.xyz'
# file = 'examples/for_paper/acrylonitrile-N2O-dihedrals_for_paper.xyz'
file = 'examples/for_paper/bifur_for_paper.xyz'
# file = 'examples/for_paper/Pummerer-gas-TS1-IRC_for_paper.xyz'

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
# files = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase'

# "New Files" to test transforming trajectories into already generated reduced dimensional space
# new_file = 'examples/SN2/SN2_dyn_BOMD.xyz'
# new_file = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase/comet-n30-traj6_fixed.xyz'
# new_file = 'examples/bifurcations/Pummerer-gas-TS1-IRC.xyz'
# new_file = 'examples/bifurcations/Pummerer_dynamics_trajectories/gas-phase/comet-n2-traj3_fixed.xyz'
new_file = 'examples/bifurcations/traj1.xyz'
# new_file = 'examples/bifurcations/traj2.xyz'
# new_file = 'examples/bifurcations/traj3.xyz'
# new_file = 'examples/bifurcations/traj4.xyz'

# Number of PCA components
ndim = 3

# 2-ala atoms
stereo_atoms_al = [6, 9, 10, 14]

# Pummerer system atoms surrounding resultant stereogenic center
stereo_atoms_P = [1, 6, 27, 32]

# Barry's bifurcation system atoms attached to allene core
stereo_atoms_B = [4, 5, 6, 7]


############################################### COORDINATES input to PCA ###############################################
# # EXAMPLE 1A: Directory as input
# traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full_coords, \
# red_coords = dim_red.dr_routine(files, ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, input_type="Coordinates")
#
# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# coords_pca_df = pd.DataFrame(coords_pca)
# colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
#           output_directory=direc, imgname=system_name, lengths=traj_lengths)
#
# # Transforming new data into generated reduced dimensional space
# dim_red.transform_new_data(new_file, direc + "/new_data", ndim, P_atom_1, P_atom_2, P_atom_3, P_atom_4, coords_pca_fit,
#                            coords_comps, coords_mean, coords_pca, traj_lengths, input_type="Coordinates")


# EXAMPLE 1B: Single file as input
# traj_lengths, system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, full, red = \
#     dim_red.dr_routine(file, ndim, input_type="Coordinates")

# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# coords_pca_df = pd.DataFrame(coords_pca)
# colorplot(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], same_axis=False, input_type="Coordinates",
#           output_directory=direc, imgname=system_name)
# colorplot_scatter(coords_pca_df[0], coords_pca_df[1], coords_pca_df[2], list(range(len(coords_pca_df[0]))))

# # Transforming new data into generated reduced dimensional space
# new_data_df = dim_red.transform_new_data(new_file, direc + "/new_data", ndim, atom_1, atom_2, atom_3, atom_4,
#                                          coords_pca_fit, coords_comps, coords_mean, coords_pca, input_type="Coordinates")
# colorplot(coords_pca_df[0], y=coords_pca_df[1], y1=coords_pca_df[2], same_axis=False, new_data=new_data_df,
#           input_type="Coordinates", output_directory=direc + "/new_data", imgname=system_name)
#
# # Barry's Bifurcation stuff
# coords_pca_df = pd.DataFrame(coords_pca)
# coords_pca_df1 = coords_pca_df[0:183]
# coords_pca_df2 = coords_pca_df.drop(coords_pca_df.index[106:184], axis=0)
# colorplot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, input_type="Coordinates", output_directory=direc, imgname=system_name)

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


# EXAMPLE 2B: Single file as input
traj_lengths1, system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values = \
   dim_red.dr_routine(file, ndim, stereo_atoms=stereo_atoms_B, input_type="Distances")

# # Plot trajectory in 2D and 3D. Only specify "lengths=" if input to dr_routine was directory
# D_pca_df = pd.DataFrame(D_pca)
# colorplot(D_pca_df[0], D_pca_df[1], D_pca_df[2], same_axis=False, input_type="Distances", output_directory=direc1,
#           imgname=(system_name1 + "_D"))
# time = list(range(len(D_pca_df[0])))
# colorplot_scatter(D_pca_df[0], D_pca_df[1], D_pca_df[2], time, name=system_name1)
#
# for point in time:
#    colorplot_scatter(D_pca_df[0], D_pca_df[1], D_pca_df[2], time, name=system_name1, point=point)


# Barry's Bifurcation stuff
D_pca_df = pd.DataFrame(D_pca)
D_pca_df1 = D_pca_df[0:183]
D_pca_df2 = D_pca_df.drop(D_pca_df.index[106:184], axis=0)
colorplot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, input_type="Distances", output_directory=direc1, imgname=(system_name1 + "_D"))



# # Print coefficients of distances that make up principal components to text files, put into output directory
# dim_red.print_distance_coeffs_to_files(direc1, ndim, system_name1, D_pca_components, atoms)
#
# # Transforming new data into generated reduced dimensional space
# new_data_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, atom_1, atom_2, atom_3, atom_4, D_pca_fit,
#                                          D_pca_components, D_mean, D_pca, input_type="Distances")
# colorplot(D_pca_df[0], y=D_pca_df[1], y1=D_pca_df[2], same_axis=False, new_data=new_data_df, input_type="Distances",
#           output_directory=direc1 + "/new_data", imgname=(system_name1 + "_D"))

# Barry's Bifurcation
new_data_df = dim_red.transform_new_data(new_file, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                         D_pca, stereo_atoms=stereo_atoms_B, input_type="Distances")
colorplot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df, input_type="Distances", output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_D_noMW_traj1"))




