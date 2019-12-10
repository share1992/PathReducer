import pandas as pd
import sys
sys.path.append('../')

import dimensionality_reduction_functions as dim_red
from plotting_functions import *


# Number of PCA components
ndim = 3

####################################### EXAMPLE 4: CYCLOPROPYLIDENE BIFURCATION ########################################
# Inputs
file = '../examples/bifurcation/bifur_IRC.xyz'
stereo_atoms_B = [3, 4, 5, 7]
# "New Files" to test transforming trajectories into already generated reduced dimensional space
# new_file1 = '../examples/bifurcation/bifur_traj1.xyz'
# new_file2 = '../examples/bifurcation/bifur_traj2.xyz'
# new_file3 = '../examples/bifurcation/bifur_traj3.xyz'
# new_file4 = '../examples/bifurcation/bifur_traj4.xyz'
new_file_exp_solv = '/Users/ec18006/Documents/Documents/CHAMPS/Explicit_solvent_MD/CHCl3_soln_1_just_solute_noN2.xyz'

# DISTANCES INPUT
system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1, aligned_original_coords, blah, blah2 = \
    dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_B, input_type="Distances")

# Transforming new data into RD space
# new_name1, new_data_df1 = dim_red.transform_new_data(new_file1, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
#                                           aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances")
# new_name2, new_data_df2 = dim_red.transform_new_data(new_file2, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
#                                           aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances")
# new_name3, new_data_df3 = dim_red.transform_new_data(new_file3, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
#                                           aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances")
# new_name4, new_data_df4 = dim_red.transform_new_data(new_file4, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
#                                           aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances")

new_name_exp, new_data_df_exp = dim_red.transform_new_data(new_file_exp_solv, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                          aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances")

# Plotting

# DISTANCES INPUT
D_pca_df = pd.DataFrame(D_pca)
D_pca_df1 = D_pca_df[0:183]
D_pca_df2 = D_pca_df.drop(D_pca_df.index[106:184], axis=0)
# colored_line_and_scatter_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
#           output_directory=direc1, imgname=(system_name1 + "_Distances_noMW"))

# colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#           output_directory=direc1, imgname=(system_name1 + "_Distances_noMW_line"))

# colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                   same_axis=False, new_data_x=new_data_df1[0], new_data_y=new_data_df1[1], new_data_z=new_data_df1[2], output_directory=direc1 + "/new_data",
#                   imgname=(system_name1 + "_Distances_noMW_traj1_D"))
# colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                   same_axis=False, new_data_x=new_data_df2[0], new_data_y=new_data_df2[1], new_data_z=new_data_df2[2], output_directory=direc1 + "/new_data",
#                   imgname=(system_name1 + "_Distances_noMW_traj2_A"))
# colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                   same_axis=False, new_data_x=new_data_df3[0], new_data_y=new_data_df3[1], new_data_z=new_data_df3[2], output_directory=direc1 + "/new_data",
#                   imgname=(system_name1 + "_Distances_noMW_traj3_B"))
# colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                   same_axis=False, new_data_x=new_data_df4[0], new_data_y=new_data_df4[1], new_data_z=new_data_df4[2], output_directory=direc1 + "/new_data",
#                   imgname=(system_name1 + "_Distances_noMW_traj4_C"))
colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
                  same_axis=False, new_data_x=new_data_df_exp[0], new_data_y=new_data_df_exp[1], new_data_z=new_data_df_exp[2], output_directory=direc1 + "/new_data")

# Plots to be used for gifs of MD data
# for i in range(len(new_data_df1[0])):
#     colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                       same_axis=False, new_data_x=new_data_df1[0], new_data_y=new_data_df1[1], new_data_z=new_data_df1[2], output_directory=direc1 + "/new_data",
#                       imgname=(system_name1 + "_Distances_noMW_traj1_D_%s" % i), points_to_circle_new_data=[i])
#
# for i in range(len(new_data_df2[0])):
#     colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                       same_axis=False, new_data_x=new_data_df2[0], new_data_y=new_data_df2[1], new_data_z=new_data_df2[2], output_directory=direc1 + "/new_data",
#                       imgname=(system_name1 + "_Distances_noMW_traj2_A_%s" % i), points_to_circle_new_data=[i])

# for i in range(len(new_data_df3[0])):
#     colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                       same_axis=False, new_data_x=new_data_df3[0], new_data_y=new_data_df3[1], new_data_z=new_data_df3[2], output_directory=direc1 + "/new_data",
#                       imgname=(system_name1 + "_Distances_noMW_traj3_B_%s" % i), points_to_circle_new_data=[i])

# for i in range(len(new_data_df4[0])):
#     colored_line_plot_projected_data(x=D_pca_df1[0], y=D_pca_df1[1], z=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], z2=D_pca_df2[2],
#                       same_axis=False, new_data_x=new_data_df4[0], new_data_y=new_data_df4[1], new_data_z=new_data_df4[2], output_directory=direc1 + "/new_data",
#                       imgname=(system_name1 + "_Distances_noMW_traj4_C_%s" % i), points_to_circle_new_data=[i])