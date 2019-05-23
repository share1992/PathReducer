import pandas as pd
import dimensionality_reduction_functions as dim_red
from plotting_functions import colored_line_plot, colored_line_and_scatter_plot

# Number of PCA components
ndim = 3

####################################### EXAMPLE 4: CYCLOPROPYLIDENE BIFURCATION ########################################
# Inputs
file = '../examples/bifurcation/bifur_IRC.xyz'
stereo_atoms_B = [3, 4, 5, 7]
# "New Files" to test transforming trajectories into already generated reduced dimensional space
new_file1 = '../examples/bifurcation/bifur_traj1.xyz'
new_file2 = '../examples/bifurcation/bifur_traj2.xyz'
new_file3 = '../examples/bifurcation/bifur_traj3.xyz'
new_file4 = '../examples/bifurcation/bifur_traj4.xyz'

# # CARTESIANS INPUT
# system_name, direc, coords_pca, coords_pca_fit, coords_comps, coords_mean, coords_vals, traj_lengths, aligned_original_traj = \
#     dim_red.pathreducer(file, ndim, input_type="Cartesians", MW=True)
#
# # Transforming new data into RD space
# new_data_df1 = dim_red.transform_new_data(new_file1, direc + "/new_data", ndim, coords_pca_fit, coords_comps,
#                                           coords_mean, aligned_original_traj, input_type="Cartesians", MW=True)
# new_data_df2 = dim_red.transform_new_data(new_file2, direc + "/new_data", ndim, coords_pca_fit, coords_comps,
#                                           coords_mean, aligned_original_traj, input_type="Cartesians", MW=True)
# new_data_df3 = dim_red.transform_new_data(new_file3, direc + "/new_data", ndim, coords_pca_fit, coords_comps,
#                                           coords_mean, aligned_original_traj, input_type="Cartesians", MW=True)
# new_data_df4 = dim_red.transform_new_data(new_file4, direc + "/new_data", ndim, coords_pca_fit, coords_comps,
#                                           coords_mean, aligned_original_traj, input_type="Cartesians", MW=True)

# DISTANCES INPUT
system_name1, direc1, D_pca, D_pca_fit, D_pca_components, D_mean, D_values, traj_lengths1, aligned_original_coords = \
    dim_red.pathreducer(file, ndim, stereo_atoms=stereo_atoms_B, input_type="Distances", MW=True)

# Transforming new data into RD space
new_data_df1 = dim_red.transform_new_data(new_file1, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                          aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances", MW=True)
new_data_df2 = dim_red.transform_new_data(new_file2, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                          aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances", MW=True)
new_data_df3 = dim_red.transform_new_data(new_file3, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                          aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances", MW=True)
new_data_df4 = dim_red.transform_new_data(new_file4, direc1 + "/new_data", ndim, D_pca_fit, D_pca_components, D_mean,
                                          aligned_original_coords, stereo_atoms=stereo_atoms_B, input_type="Distances", MW=True)


# Plotting
# # CARTESIANS INPUT
# coords_pca_df = pd.DataFrame(coords_pca)
# coords_pca_df1 = coords_pca_df[0:183]
# coords_pca_df2 = coords_pca_df.drop(coords_pca_df.index[106:184], axis=0)
# colored_line_plot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, output_directory=direc, imgname=(system_name + "_Cartesians_MW"))
# colored_line_plot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, new_data=new_data_df1, output_directory=direc + "/new_data",
#           imgname=(system_name + "_Cartesians_MW_line_traj1"))
# colored_line_plot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, new_data=new_data_df2, output_directory=direc + "/new_data",
#           imgname=(system_name + "_Cartesians_MW_line_traj2"))
# colored_line_plot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, new_data=new_data_df3, output_directory=direc + "/new_data",
#           imgname=(system_name + "_Cartesians_MW_line_traj3"))
# colored_line_plot(coords_pca_df1[0], y=coords_pca_df1[1], y1=coords_pca_df1[2], x2=coords_pca_df2[0], y2=coords_pca_df2[1],
#           y12=coords_pca_df2[2], same_axis=False, new_data=new_data_df4, output_directory=direc + "/new_data",
#           imgname=(system_name + "_Cartesians_MW_line_traj4"))


# DISTANCES INPUT
D_pca_df = pd.DataFrame(D_pca)
D_pca_df1 = D_pca_df[0:183]
D_pca_df2 = D_pca_df.drop(D_pca_df.index[106:184], axis=0)
colored_line_and_scatter_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, output_directory=direc1, imgname=(system_name1 + "_Distances_MW"))

colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df1, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_Distances_MW_traj1_D"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df2, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_Distances_MW_traj2_A"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df3, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_Distances_MW_traj3_B"))
colored_line_plot(D_pca_df1[0], y=D_pca_df1[1], y1=D_pca_df1[2], x2=D_pca_df2[0], y2=D_pca_df2[1], y12=D_pca_df2[2],
          same_axis=False, new_data=new_data_df4, output_directory=direc1 + "/new_data",
          imgname=(system_name1 + "_Distances_MW_traj4_C"))
